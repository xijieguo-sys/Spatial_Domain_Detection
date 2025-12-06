import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv
from torch.nn import BatchNorm1d
from torch.autograd import Function
logger = logging.getLogger(__name__)


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class GNNLayers(nn.Module):
    def __init__(self, layer_dim, dropout, norm='batchnorm', activation='relu', last_norm=False):
        super().__init__()
        self.layer_dim = layer_dim
        self.num_layers = len(layer_dim) - 1
        self.gnn_layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.norm = norm
        self.activation = create_activation(activation)
        self.norm_layers = torch.nn.ModuleList()  # For batch normalization layers
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                if last_norm:
                    self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))
                else:
                    self.norm_layers.append(nn.Identity())
            else:
                self.norm_layers.append(create_norm(self.norm)(layer_dim[i + 1]))

    def forward(self, inputs, edge_index):
        h = inputs
        for l in range(self.num_layers):
            h = self.gnn_layers[l](h, edge_index)
            h = self.norm_layers[l](h)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GCNLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        for i in range(self.num_layers):
            self.gnn_layers.append(GCNConv(layer_dim[i], layer_dim[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, GINConv):
                # Manual Kaiming initialization for MLP inside GIN
                for m in layer.nn:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


class GATLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm, heads=4):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        for i in range(self.num_layers):
            self.gnn_layers.append(GATConv(layer_dim[i], layer_dim[i + 1] // heads, heads=heads))
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, GINConv):
                # Manual Kaiming initialization for MLP inside GIN
                for m in layer.nn:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


class GINLayers(GNNLayers):
    def __init__(self, layer_dim, dropout, norm, activation, last_norm):
        super().__init__(layer_dim=layer_dim, dropout=dropout, norm=norm, activation=activation, last_norm=last_norm)
        for i in range(self.num_layers):
            mlp = torch.nn.Sequential(
                ## first map to intermediate, then map to output
                torch.nn.Linear(layer_dim[i], (layer_dim[i] + layer_dim[i + 1]) // 2),
                create_norm(norm)((layer_dim[i] + layer_dim[i + 1]) // 2),
                self.activation,
                torch.nn.Linear((layer_dim[i] + layer_dim[i + 1]) // 2, layer_dim[i + 1]),
                create_norm(norm)(layer_dim[i + 1]),
                self.activation,
            )
            self.gnn_layers.append(GINConv(mlp))

        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, GINConv):
                # Manual Kaiming initialization for MLP inside GIN
                for m in layer.nn:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


class NodeTransform(nn.Module):
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(NodeTransform, self).__init__()
        self.mlp = mlp

        if norm not in ["layernorm", "batchnorm"]:
            self.norm = nn.Identity()
        else:
            norm_func = create_norm(norm)
            self.norm = norm_func(self.mlp.output_dim)

        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h
    

class MLP(nn.Module):
    def __init__(self, layer_dim=[], norm="batchnorm", activation="relu"):
        super().__init__()
        assert len(layer_dim) >= 2, f"MLP layer_dim={layer_dim}, at least specify input & output dim!"
        self.num_layers = len(layer_dim) - 1
        self.output_dim = layer_dim[-1]
        if self.num_layers == 1:  # linear
            self.linear = nn.Linear(layer_dim[0], layer_dim[1])
        else:  # non-linear
            self.linear_layers = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            for l in range(self.num_layers):
                self.linear_layers.append(nn.Linear(layer_dim[l], layer_dim[l + 1]))
                if l != self.num_layers - 1:
                    self.norms.append(create_norm(norm)(layer_dim[l + 1]))
                    self.activations.append(create_activation(activation))

        self.reset_parameters()

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            for i in range(self.num_layers - 1):
                x = self.linear_layers[i](x)
                x = self.norms[i](x)
                x = self.activations[i](x)
            return self.linear_layers[-1](x)

    def reset_parameters(self):
        if self.num_layers == 1:
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        else:
            for layer in self.linear_layers:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class BilinearDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super(BilinearDiscriminator, self).__init__()
        self.bilinear = nn.Bilinear(in_dim, in_dim, 1, bias=True)
        torch.nn.init.xavier_normal_(self.bilinear.weight.data)
        self.bilinear.bias.data.fill_(0.0)

    def forward(self, x, x_contrast):
        logits = self.bilinear(x, x_contrast)  # no softmax here if using BCEWithLogitsLoss
        return logits


    
class LinearLayers(nn.Module):
    def __init__(self, layer_dim=[], norm='layernorm', activation='relu'):
        super().__init__()
        self.output = MLP(layer_dim=layer_dim, norm=norm, activation=activation)

    def forward(self, x):

        return self.output(x)
        


class CellSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, norm='layernorm'):
        """
        input_dim: number of genes
        embed_dim: embedding size (you can set this smaller than input_dim)
        """
        super(CellSelfAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.key_proj = nn.Linear(input_dim, embed_dim)
        self.value_proj = nn.Linear(input_dim, embed_dim)
        self.scale = embed_dim ** 0.5  
        self.norm = create_norm(norm)

        self.reset_parameters()

    def forward(self, x):
        """
        x: [num_spots, num_genes] input matrix
        Returns:
            embeddings: [num_spots, embed_dim]
            attention_scores: [num_spots, num_spots]
        """
        Q = self.query_proj(x)  
        K = self.key_proj(x)   
        V = self.value_proj(x) 

        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale  
        attn_probs = F.softmax(attn_scores, dim=-1)  
        embeddings = torch.matmul(attn_probs, V)
        outputs = self.norm(embeddings)

        return outputs, attn_probs

    def reset_parameters(self):
        for proj in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.kaiming_normal_(proj.weight, nonlinearity='relu')
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

# class NicheAttention(nn.Module):
#     def __init__(self, input_dim, embed_dim, norm='layernorm', activation='relu'):
#         """
#         input_dim: number of genes
#         embed_dim: embedding size (you can set this smaller than input_dim)
#         """
#         super(NicheAttention, self).__init__()
#         self.cell_att = CellSelfAttention(input_dim=input_dim, embed_dim=embed_dim)
#         self.norm = create_norm(norm)
#         self.mlp = LinearLayers(layer_dim=[embed_dim, embed_dim], norm=norm, activation=activation)


#     def forward(self, x):
        


