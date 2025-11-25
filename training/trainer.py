import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import glob
import scanpy as sc
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch, gc
import pytorch_lightning as pl

from collections import Counter

from dataset import preprocess, rank_genes_per_cell, SpatialDataset, spatial_collate_fn
from model import SpatialDomainModel



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



class Bioformer_trainer:
    def __init__(self, args):
        self.args = args
        self.epochs = self.args.epochs

    
    def main_train(self, args):
        # all_results = {}
        test_file_name = os.path.basename(args.data_path).replace('.h5ad', '')
        test_ari_file = 'test_ari.json'
        args.result_specific_folder = os.path.join(args.result_general_folder, test_file_name)
        test_ari_path = os.path.join(args.result_specific_folder, test_ari_file)
        os.makedirs(args.result_specific_folder, exist_ok=True)
      

        # set_seed(args.other_random_state)
        pl.seed_everything(args.other_random_state, workers=True)
        
        os.environ["PYTHONHASHSEED"] = str(args.other_random_state)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
            
        torch.manual_seed(args.init_seed)
        torch.cuda.manual_seed(args.init_seed)       
        torch.cuda.manual_seed_all(args.init_seed)

        
        test_ari = self.train(args)

        test_dict = {
            test_file_name: float(test_ari)    # make sure it's JSON-serializable
        }
        with open(test_ari_path, 'w') as f:
            json.dump(test_dict, f, indent=4)


    def train(self, args):
        adata = sc.read_h5ad(args.data_path)
        adata = preprocess(adata)
        # gene_names, _ = rank_genes_per_cell(adata)
        if str(adata.obs[args.annot_label].dtype) == "category":
            y = adata.obs[args.annot_label].cat.codes.to_numpy()
        else:
            # convert anything to category codes
            y = adata.obs[args.annot_label].astype("category").cat.codes.to_numpy()
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        dataset = SpatialDataset(X, y=y, min_expr=0.0)
        num_genes = dataset.n_genes

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,        # ← choose batch size
            shuffle=True,          # ← shuffle ON for training
            collate_fn=spatial_collate_fn,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,             # validation: fixed order for alignment
            collate_fn=spatial_collate_fn
        )
        test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,            # IMPORTANT for alignment
            collate_fn=spatial_collate_fn,
        )

        model = SpatialDomainModel(
            args,
            num_genes=num_genes,
            cell_dim=args.cell_dim,
            trans_heads=args.trans_heads,
            trans_layer=args.trans_layer,
            trans_ffd=args.trans_ffd,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.wd
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.result_specific_folder, 
            filename="epoch{epoch}-train_acc{train_slice_accuracy:.4f}",
            monitor="train_slice_accuracy",                     
            mode="max",   
            save_top_k = 0,                                                           
            save_last=False                           
        )

        # csv_logger = CSVLogger(
        #     save_dir=args.fold_folder,
        #     name = 'csv_logs'
        #       # Parent folder
        #             # Optional: sub-subfolder (or auto-increments if not set)
        # )

        tb_logger = TensorBoardLogger(
            save_dir=args.result_specific_folder,
            name = 'tb_logs'
        )

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=args.epochs,
            logger=[tb_logger],
            deterministic=True,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            num_sanity_val_steps=0
        )

        gc.collect()
        torch.cuda.empty_cache()
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
        test_ari = model.test_ari_


        return test_ari

