import torch_geometric.seed
import torch
import os
import pickle
import random
import logging
import anndata
import numpy as np
import scanpy as sc
import squidpy as sq
import networkx as nx
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def get_feature(adata, param):
    device = param['device']
    print(adata)
    if not param['pca']:
        feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
        return feature
    else:
        feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
        return feature


def prepare_graph_data(adata_ref_list, param):
    logger.info(f"constructing spatial graph, computing {param['model_k']}-hop subgraph and creating batch labels for each sample!")
    feature_list, edge_ind_list = [], []
    sub_node_sample_list, sub_edge_ind_sample_list = [], [] 
    for i in range(0,len(adata_ref_list)):
        # feature = get_feature(adata_ref_list[i], query=False, param=param, ref_id=adata_ref_list[i].uns['library_id'], device=param['device'])
        #########################
        sub_node_save = []
        sub_edge_save = []
        #########################
        print(adata_ref_list[i])
        feature = get_feature(adata_ref_list[i], param=param)

        ## so spatial connectivities become a adjacency graph now
        adj_mat = adata_ref_list[i].obsp['spatial_connectivities'].tocoo()
        edge_index = torch.tensor(np.vstack((adj_mat.row, adj_mat.col)), dtype=torch.int64).to(param['device']) 

        assert edge_index.min() >= 0, f"Negative index found in edge_index: {edge_index.min()}"
        assert edge_index.max() < adata_ref_list[i].n_obs, f"Index out of bounds: max {edge_index.max()} >= {adata_ref_list[i].n_obs}"

        k = param['model_k']

        logger.info(f"computing {k} hop subgraph for sample {adata_ref_list[i].uns['file_name']}")
        sub_node_list, sub_edge_ind_list = [], []
        ## n_obs: number of cells
        ## for each cell in a sample, generate k-hop subgraph
        for node_ind in tqdm(range(adata_ref_list[i].n_obs)):
            ## sub_nodes: list of nodes in the k-hop neighborhood
            ## subgraph's edge list

            assert node_ind < adata_ref_list[i].n_obs, f"node_ind {node_ind} is out of range!"
            sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(node_ind, k, edge_index, relabel_nodes=True)

            ###################################
            # sub_node_save.append(sub_nodes.cpu().numpy())
            # sub_edge_save.append(sub_edge_index.cpu().numpy())
            ###################################

            sub_node_list.append(sub_nodes)
            sub_edge_ind_list.append(sub_edge_index)
        
        # save_path = os.path.join(graph_save_path, adata_ref_list[i].uns['file_name'].replace('.h5ad', '.pkl'))
        # with open(save_path, "wb") as f:
        #     pickle.dump({"nodes": sub_node_save, "edges": sub_edge_save}, f)

        feature_list.append(feature)
        edge_ind_list.append(edge_index)
        sub_node_sample_list.append(sub_node_list)
        sub_edge_ind_sample_list.append(sub_edge_ind_list)
        

    return feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list


def fix_var_index_conflict(adata):
    if adata.var.index.name in adata.var.columns:
        if not (adata.var.index.to_series().equals(adata.var[adata.var.index.name])):
            adata.var.index.name = None
        else:
            adata.var = adata.var.drop(columns=[adata.var.index.name])

def preprocess(args, adata, save_data=True, target_sum=1e4):

    # select hvg for homogenes of each dataset separately then take the intersect
    # print(adata)
    
    sq.gr.spatial_neighbors(adata, coord_type=args.graph_build, n_neighs=args.knn)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hvg)
    adata = adata[:, adata.var['highly_variable']]

    # normalization the ref homo
    adata.var_names_make_unique()
    # adata_raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=True, max_value=None, copy=False)

    # if save_data:
    #     fix_var_index_conflict(adata)
    #     fix_var_index_conflict(adata_raw)
    #     adata.raw = adata_raw
        
    #     # TODO: later modify it to save in the original dataset (or somewhere else) instead of the output files so that no need to repeat preprocessing for different runs
    #     adata.write_h5ad(f'{args.savedir}/adata_preprocessed.h5ad')
    
    return adata.copy()
        

def filter_adata(adata):
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True, log1p=False)
    # percent_mito = 20
    # filter_mincells = 3
    # filter_mingenes = 100

    # adata = adata[adata.obs["n_genes_by_counts"] > filter_mingenes, :].copy()
    # adata = adata[adata.obs["pct_counts_mt"] < percent_mito, :].copy()
    # adata = adata[:, adata.var["n_cells_by_counts"] > filter_mincells].copy()

    return adata