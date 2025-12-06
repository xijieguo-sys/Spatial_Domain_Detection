import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import ClusterData, ClusterLoader
from tqdm import tqdm
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from model import NicheST
from plot import plot_separate_loss_curves



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



class NicheST_trainer:
    def __init__(self, in_dim, param):
        self.param = param
        self.model = NicheST(in_dim, param)
        self.model = self.model.to(self.param['device'])
        self.optimizer = self.model.build_optimizer()
        self.epochs = self.param['epochs']
        
        

    def train_epoch(self, epoch, feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list):
        self.model.train()
        loss_sum, recon_loss_sum, contra_pos_loss_sum, contra_neg_loss_sum = 0.0, 0.0, 0.0, 0.0

        num_slices = len(feature_list)
        
        for idx in range(num_slices):
            feature_list[idx] = feature_list[idx].to(self.param['device'])
            edge_ind_list[idx] = edge_ind_list[idx].to(self.param['device'])
            # sub_node_sample_list[idx] = sub_node_sample_list[idx].to(self.param['device'])
            # sub_edge_ind_sample_list[idx] = sub_edge_ind_sample_list[idx].to(self.param['device'])

            self.optimizer.zero_grad()
            recon, logits_pos, logits_neg = self.model(feature_list[idx], edge_ind_list[idx], 
                                                       sub_node_sample_list[idx], sub_edge_ind_sample_list[idx])
            
            loss, recon_loss, contra_pos_loss, contra_neg_loss = self.model.compute_loss(feature_list[idx], recon, logits_pos, logits_neg)

            loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            contra_pos_loss_sum += contra_pos_loss.item()
            contra_neg_loss_sum += contra_neg_loss.item()

            loss.backward()
            self.optimizer.step()

        final_loss = loss_sum / num_slices
        final_recon_loss = (recon_loss_sum / num_slices)*self.param['recon_weight']
        final_contra_pos_loss = (contra_pos_loss_sum / num_slices)*self.param['contra_pos_weight']
        final_contra_neg_loss = (contra_neg_loss_sum / num_slices)*self.param['contra_neg_weight']
        
        print(f'Epoch {epoch}:')
        print('Total Loss: {:.6f}'.format(final_loss))
        print('Reconstruction Loss: {:.6f}'.format(final_recon_loss))
        print('Contrastive Pos Loss: {:.6f}'.format(final_contra_pos_loss))
        print('Contrastive Neg Loss: {:.6f}'.format(final_contra_neg_loss))

        return final_loss, final_recon_loss, final_contra_pos_loss, final_contra_neg_loss
       
    def train(self, feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list):
        total_losses = []
        recon_losses = []
        contra_pos_losses = []
        contra_neg_losses = []

        best_total_loss = 1e5
        
        for epoch in range(self.epochs):
            final_loss, final_recon_loss, final_contra_pos_loss, final_contra_neg_loss = self.train_epoch(epoch, 
                            feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list)
            total_losses.append(final_loss)
            recon_losses.append(final_recon_loss)
            contra_pos_losses.append(final_contra_pos_loss)
            contra_neg_losses.append(final_contra_neg_loss)

            # if final_recon_loss < best_total_loss:
            #     best_total_loss = final_loss
            #     torch.save(self.model.state_dict(), os.path.join(self.param['savedir'], f'model_weights_best_loss_{epoch}.pth'))

            # if epoch % 100 == 0:
            #     torch.save(self.model.state_dict(), os.path.join(self.param['savedir'], f'model_weights_{epoch}.pth'))
        torch.save(self.model.state_dict(), os.path.join(self.param['savedir'], f'model_weights_{epoch}.pth'))
        plot_dir = os.path.join(self.param['savedir'], 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        plot_separate_loss_curves(total_losses, recon_losses, contra_pos_losses, contra_neg_losses, plot_dir)

