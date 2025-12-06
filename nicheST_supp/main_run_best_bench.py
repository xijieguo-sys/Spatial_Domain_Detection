import argparse
import numpy as np
import os
import torch
import pickle 
from preprocess import *
import random
import time
from train import NicheST_trainer
import optuna
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from pandas.api.types import CategoricalDtype
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from plot import plot_ari_clustering
import ot

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


def downstream(args):
    
    set_seed(args.seed)
    param = vars(args)
    args.savedir = os.path.join(args.save_dir, f'drop{args.dropout:.3f}_knn{args.knn}_seed{args.seed}_dims{len(args.enc_dims)}_{args.enc_dims[0]}_decdims{len(args.dec_dims)}_pos{args.contra_pos_weight:.3f}_neg{args.contra_neg_weight:.3f}_{args.gnn}_k{args.model_k}_declinear')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    print(f"The saving directory set to {args.savedir}", flush=True)
    adata = sc.read_h5ad(args.adata_path)
    adata = filter_adata(adata)
    adata = preprocess(args, adata)

    feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list = prepare_graph_data([adata], param)
    nicheST_trainer = NicheST_trainer(feature_list[0].shape[1], param)
    nicheST_trainer.train(feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list)

    num_layers = adata.obs[args.orig_layer].nunique()
    predicted_X = nicheST_trainer.model.generate_embedding(feature_list[0], edge_ind_list[0],
                                                           sub_node_sample_list[0], sub_edge_ind_sample_list[0])
    
    X_numpy = predicted_X.detach().cpu().numpy()

    np.save(os.path.join(args.savedir, 'predicted_X.npz'), X_numpy)

    adata.obsm[args.embed_type] = X_numpy
    pca = PCA(n_components=20, random_state=args.seed)
    X_pca = pca.fit_transform(X_numpy)
    

    gmm = GaussianMixture(n_components=num_layers, covariance_type='tied', random_state=args.seed)
    gmm.fit(X_pca)
    cluster_labels = gmm.predict(X_pca)


    adata.obs[args.predicted_layer] = cluster_labels
    np.save(os.path.join(args.savedir, 'cluster_labels.npy'), cluster_labels)

    
    new_label = refine_label(adata, args.radius, args.predicted_layer)
    adata.obs[args.refine_label] = new_label
    # adata.write(os.path.join(args.savedir, 'adata_labeled.h5ad'))
    np.save(os.path.join(args.savedir, 'refined_labels.npy'), new_label)


    adata = adata[~pd.isnull(adata.obs[args.orig_layer])]
    true_labels_cat = pd.Categorical(adata.obs[args.orig_layer])
    pred_labels_cat = pd.Categorical(adata.obs[args.refine_label])
    true_color_indices = true_labels_cat.codes
    pred_color_indices = pred_labels_cat.codes
    all_categories = true_labels_cat.categories
    ari = adjusted_rand_score(true_color_indices, pred_color_indices)
    plot_ari_clustering(adata, true_color_indices, pred_color_indices, all_categories, ari, args.savedir)
    new_dir = args.savedir + f'_ari{ari:.3f}'
    if os.path.exists(args.savedir) and not os.path.exists(new_dir):
        os.rename(args.savedir, new_dir)

    return ari


def main():
    parser = argparse.ArgumentParser(description='NicheST with Optuna')
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--adata_path', type=str, default=None)

    parser.add_argument('--att_heads', type=int, default=1)
    ## didn't care about preprocess, preprocessed anyway
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--downstream', type=str, default=None)
    parser.add_argument('--knn', type=int, default=6)
    parser.add_argument('--pca', type=bool, default=False)
    parser.add_argument('--graph_build', type=str, default='grid')
    parser.add_argument('--model_k', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train', action='store_true')

    parser.add_argument('--hvg', type=int, default=3000)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--orig_layer', type=str, default='layer')
    parser.add_argument('--predicted_layer', type=str, default='predicted_layer')

    parser.add_argument('--refine_label', type=str, default='refine_label')


    #### model related ####
    parser.add_argument('--clustered', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    parser.add_argument('--dropout', type=float, default=0.1)


    ##### these two need careful treatment when passing to command line
    # parser.add_argument('--enc_dims', type=list, default=[128, 128])
    # parser.add_argument('--dec_dims', type=list, default=[128])
    parser.add_argument('--enc_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--dec_dims', type=int, nargs='+', default=[128])
    ########

    parser.add_argument('--contra_pos_weight', type=float, default=1.0)
    parser.add_argument('--contra_neg_weight', type=float, default=1.0)
    parser.add_argument('--gnn', type=str, default='gcn')

    parser.add_argument('--att_thresh', type=float, default=1e-3)
    parser.add_argument('--recon_weight', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--radius', type=int, default=50)

    parser.add_argument('--contra_type', type=str, default='mine')
    parser.add_argument('--recon_loss_type', type=str, default='mse')
    parser.add_argument('--norm', type=str, default='layernorm')
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--dec_type', type=str, default='linear')

    parser.add_argument('--embed_type', type=str, default='recon')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # set_seed(args.seed)
    

    # if args.preprocess:
    #     adata = sc.read_h5ad(args.adata_path)
    #     adata = filter_adata(adata)
    #     adata = preprocess(args, adata)
    # else:
    #     adata = sc.read_h5ad(os.path.join(args.savedir, 'adata_preprocessed.h5ad'))

    # sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    # study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.optimize(lambda trial: objective(trial, args), n_trials=1000)

    # print("Best ARI score:", study.best_value)
    # print("Best hyperparameters:", study.best_params)


    ari = downstream(args)
    print('Best ARI score: ', ari)



if __name__ == '__main__':
    main()