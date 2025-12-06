import logging
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from layers import GINLayers, BilinearDiscriminator, GCNLayers, GATLayers, CellSelfAttention, LinearLayers
from torch_geometric.nn.pool import global_mean_pool
import bench_utils as bench_utils
logger = logging.getLogger(__name__)

def cos_sim_mean(x1, x2):
    return 1 - F.cosine_similarity(x1, x2).mean()

def create_recon(name):
    if name == 'mse':
        return F.mse_loss
    elif name == 'cos':
        return cos_sim_mean
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class NicheST(nn.Module):
    def __init__(self, in_dim, param=None, logger=None):
        super().__init__()
        # self.logger = logger
        # self.logger.info(f"building model, in dim={in_dim}, time: {bench_utils.get_time_str()}")
        self.encoder = None
        self.decoder = None
        self.param = param

        self.gnn = param['gnn']
        self.contra_type = param['contra_type']
        self.att_heads = param['att_heads']
        self.avg_att = False
        if self.att_heads > 1:
            self.avg_att = True

        self.att_thresh = param['att_thresh']
        
        self.recon_loss_type = param['recon_loss_type']
        self.recon_weight = param['recon_weight']
        self.contra_pos_weight = param['contra_pos_weight']
        self.contra_neg_weight = param['contra_neg_weight']

        assert param['enc_dims'][-1] == param['dec_dims'][0], f"encoder {param['enc_dims']} and decoder {param['dec_dims']} latent dimension does not match!"
        self.enc_dims = [in_dim] + param['enc_dims']
        self.att_dim = param['enc_dims'][-1]
        self.zdim = param['enc_dims'][-1]
        self.dec_dims = param['dec_dims'] + [in_dim]

        self.lr = param['lr']
        self.norm = param['norm']
        self.dropout = param['dropout']
        self.activation = param['activation']
        self.weight_decay = param['weight_decay']

        self.build_model()

    def low_att_sampling(self, attn, ratio=0.25):
        num_spots = attn.size(0)
        k = int(num_spots * ratio)
        values, indices = torch.topk(attn, k, dim=-1, largest=False, sorted=False)

        rand_idx = torch.randint(0, k, (num_spots,), device=attn.device)
        sampled_indices = indices[torch.arange(num_spots), rand_idx]

        return sampled_indices
    
    def get_negative_global(self, global_niches, sampled_indices):
        return global_niches[sampled_indices]
        
        
    def get_local_neighbor(self, x, sub_node_list, sub_edge_list):
        subgraph_list = [Data(x=x[sub_node_list[node_ind]], edge_index=sub_edge_list[node_ind]) for node_ind in range(x.shape[0])]
        subgraph_batch = Batch.from_data_list(subgraph_list)
        z = global_mean_pool(subgraph_batch.x, subgraph_batch.batch)
        # print(x.shape)
        # print(z.shape)
        return z


    def encode_subgraph(self, x, edge_index, subgraph_mask):
        """receives whole sample feature and subgraph mask, returns representation of target region"""
        x = self.encoder(x, edge_index)
        sub_x = x[subgraph_mask]
        sub_z = torch.mean(sub_x, dim=0)
        return sub_z

    def create_gnn(self, name, dims, dropout, norm, activation, last_norm):
        if name == 'gcn':
            return GCNLayers(layer_dim=dims, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        elif name == 'gat':
            return GATLayers(layer_dim=dims, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        elif name == 'gin':
            return GINLayers(layer_dim=dims, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)

    def reset_multihead_attention(self, attn: nn.MultiheadAttention):
        # Initialize Q, K, V combined weight
        nn.init.kaiming_normal_(attn.in_proj_weight, nonlinearity='relu')
        if attn.in_proj_bias is not None:
            nn.init.zeros_(attn.in_proj_bias)
        
        # Initialize the output projection
        nn.init.kaiming_normal_(attn.out_proj.weight, nonlinearity='relu')
        if attn.out_proj.bias is not None:
            nn.init.zeros_(attn.out_proj.bias)
        
    
    def build_model(self):
        ## has encoding dimensions here
        # self.encoder = GINLayers(layer_dim=self.enc_dims, dropout=self.dropout, norm=self.norm, activation=self.activation, last_norm=True)
        self.encoder = self.create_gnn(self.gnn, self.enc_dims, self.dropout, self.norm, self.activation, True)
        # self.dec_dims[0] += self.dec_batch_dim
        if self.param['dec_type'] == 'graph':
            self.decoder = self.create_gnn(self.gnn, self.dec_dims, self.dropout, self.norm, self.activation, False)
        elif self.param['dec_type'] == 'linear':
            self.decoder = LinearLayers(self.dec_dims, self.norm, self.activation)
        ## could do mutlihead attention and average attention weights
        self.attention = nn.MultiheadAttention(self.att_dim, self.att_heads, self.dropout, batch_first=True)
        self.reset_multihead_attention(self.attention)
        self.contrastive_discriminator = BilinearDiscriminator(in_dim=self.zdim)
        
        self.contra_criterion = F.binary_cross_entropy_with_logits
        self.recon_criterion = create_recon(self.recon_loss_type)
      

    def build_optimizer(self):
        param_list = (list(self.encoder.parameters()) +
                      list(self.contrastive_discriminator.parameters()) + list(self.decoder.parameters()) + 
                      list(self.attention.parameters()))
        optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    
    def compute_loss(self, x, recon, logits_pos, logits_neg):
        recon_loss = self.recon_criterion(recon, x)
        contra_pos_loss = self.contra_criterion(logits_pos, torch.ones_like(logits_pos))
        contra_neg_loss = self.contra_criterion(logits_neg, torch.zeros_like(logits_neg))

        loss = self.recon_weight * recon_loss + self.contra_pos_weight * contra_pos_loss + \
                self.contra_neg_weight * contra_neg_loss


        return loss, recon_loss, contra_pos_loss, contra_neg_loss


    def forward(self, x, edge_index, sub_node_list, sub_edge_list):
        spot_emb = self.encoder(x, edge_index)
        local_niches = self.get_local_neighbor(spot_emb, sub_node_list, sub_edge_list)

        if self.param['dec_type'] == 'graph':
            recon = self.decoder(local_niches, edge_index)
        elif self.param['dec_type'] == 'linear':
            recon = self.decoder(local_niches)
        
        local_niches = local_niches.unsqueeze(0)
        global_niches, attn_weights = self.attention(local_niches, local_niches, local_niches, 
                                                     need_weights=True, average_attn_weights=self.avg_att)
        local_niches = local_niches.squeeze(0)
        global_niches = global_niches.squeeze(0)
        attn_weights = attn_weights.squeeze(0)

        if not self.avg_att:
            attn_weights = attn_weights.squeeze(0)
        
        sampled_indices = self.low_att_sampling(attn_weights, self.att_thresh)
        negative_global_niches = self.get_negative_global(global_niches, sampled_indices)

        if self.contra_type == 'mine':
            logits_pos = self.contrastive_discriminator(local_niches, global_niches)
            logits_neg = self.contrastive_discriminator(local_niches, negative_global_niches)
        
        return recon, logits_pos, logits_neg
    

    @ torch.no_grad()
    def generate_embedding(self, x, edge_index, sub_node_list, sub_edge_list):

        self.eval()
        spot_emb = self.encoder(x, edge_index)
        local_niches = self.get_local_neighbor(spot_emb, sub_node_list, sub_edge_list)

        if self.param['embed_type'] == 'recon':
            if self.param['dec_type'] == 'graph':   
                recon = self.decoder(local_niches, edge_index)
            elif self.param['dec_type'] == 'linear':
                recon = self.decoder(local_niches)
            return recon
        elif self.param['embed_type'] == 'local':
            return local_niches
        elif self.param['embed_type'] == 'spot':
            return spot_emb


    

    
    
