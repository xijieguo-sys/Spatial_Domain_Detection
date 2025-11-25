import argparse
import numpy as np
import os
import torch
import pickle 
import random
import time
from trainer import Bioformer_trainer
import pandas as pd
import scanpy as sc
import argparse



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




def get_args():
    parser = argparse.ArgumentParser(description="NicheST Training Configuration")

    # General training
    parser.add_argument('--epochs', type=int, default=60, help='Number of max training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', '--weight_decay', type=float, default=0, dest='wd', help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size during training')
    parser.add_argument('--dropout', type=float, default=0.)
    # Cross-validation
    parser.add_argument('--other_random_state', type=int, default=42, help='Random seed for other randomness')
    parser.add_argument('--init_seed', type=int, default=42, help='Seed for model initialization')

    # Paths
    parser.add_argument('--data_path', type=str, required=True, help='Directory containing dataset')
    parser.add_argument('--result_general_folder', type=str, required=True, help='Folder to store results')
    parser.add_argument('--annot_label', type=str, required=True)

    # Model dimensions
    parser.add_argument('--hvg_num', type=int, default=3000)
    parser.add_argument('--cell_dim', type=int, default=256)
    parser.add_argument('--trans_layer', type=int, default=2)
    parser.add_argument('--trans_ffd', type=int, default=1024)
    parser.add_argument('--trans_heads', type=int, default=1)

    parser.add_argument('--dec_dims', type=int, nargs='*', default=[])

    parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight for reconstruction loss')

   
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    trainer = Bioformer_trainer(args)
    trainer.main_train(args)


if __name__ == '__main__':
    main()