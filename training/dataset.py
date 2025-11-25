import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import scanpy as sc
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import issparse

class SpatialDataset(Dataset):
    """
    X: np.ndarray or torch.Tensor of shape (n_cells, n_genes)
       raw or normalized expression matrix
    y: array-like of shape (n_cells,) with ground-truth labels (e.g., spatial domains)
       can be int or categorical codes.
    """
    def __init__(self, X, y=None, min_expr=0.0):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.X = X
        self.min_expr = min_expr
        self.n_cells, self.n_genes = X.shape

        if y is None:
            self.y = None
        else:
            y = np.asarray(y)
            # store as long tensor
            self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        expr = self.X[idx]  # (n_genes,)

        # 1) mask out low / zero-expressed genes
        mask = expr > self.min_expr
        expr_nonzero = expr[mask]               # (n_nonzero,)
        gene_indices = torch.arange(self.n_genes)[mask]

        if expr_nonzero.numel() == 0:
            # degenerate: no expressed genes -> dummy sequence
            ranked_gene_ids = torch.tensor([0], dtype=torch.long)
        else:
            # 2) sort genes by expression (descending)
            sorted_idx = torch.argsort(expr_nonzero, descending=True)
            ranked_gene_ids = gene_indices[sorted_idx].long()

        label = -1
        if self.y is not None:
            label = self.y[idx]

        return ranked_gene_ids, expr, label

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata

def rank_genes_per_cell(adata):
    """
    For each cell, rank genes by expression (descending),
    excluding genes with 0 expression in that cell.

    Parameters
    ----------
    adata : AnnData
        Assumes adata.X contains the expression you care about
        (e.g. log-normalized counts, and possibly only HVGs).
    top_k : int or None
        If not None, only keep the top_k genes per cell.

    Returns
    -------
    ranked_genes_per_cell : list of np.ndarray (str)
        ranked_genes_per_cell[i] is an array of gene names for cell i,
        sorted from highest to lowest expression.
    ranked_values_per_cell : list of np.ndarray (float)
        Same shape as ranked_genes_per_cell, with expression values.
    """
    X = adata.X
    n_cells, n_genes = X.shape
    gene_names = np.array(adata.var_names)

    ranked_genes_per_cell = []
    ranked_values_per_cell = []

    if issparse(X):
        X = X.tocsr()
        for i in range(n_cells):
            row = X[i].toarray().ravel()
            ## only keep non-zero expression genes
            mask = row > 0
            vals = row[mask]
            genes = gene_names[mask]

            if vals.size == 0:
                ranked_genes_per_cell.append(np.array([], dtype=str))
                ranked_values_per_cell.append(np.array([], dtype=float))
                continue

            order = np.argsort(-vals)  # descending

            ranked_genes_per_cell.append(genes[order])
            ranked_values_per_cell.append(vals[order])
    else:
        for i in range(n_cells):
            row = np.asarray(X[i]).ravel()
            ## only keep non-zero expression genes
            mask = row > 0
            vals = row[mask]
            genes = gene_names[mask]

            if vals.size == 0:
                ranked_genes_per_cell.append(np.array([], dtype=str))
                ranked_values_per_cell.append(np.array([], dtype=float))
                continue

            order = np.argsort(-vals)

            ranked_genes_per_cell.append(genes[order])
            ranked_values_per_cell.append(vals[order])

    return ranked_genes_per_cell, ranked_values_per_cell

# def spatial_collate_fn(batch):
#     """
#     batch: list of (ranked_gene_ids, expr)
#            ranked_gene_ids: LongTensor [L_i]
#            expr: FloatTensor [num_genes]
#     Returns:
#       {
#         "input_ids": LongTensor (B, L_max),
#         "attn_mask": BoolTensor (B, L_max),
#         "expr":      FloatTensor (B, num_genes)
#       }
#     """
#     ranked_list, expr_list = zip(*batch)  # tuples of tensors

#     # pad sequences
#     lengths = [x.size(0) for x in ranked_list]
#     input_ids = pad_sequence(ranked_list, batch_first=True, padding_value=0)  # (B, L_max)

#     B, L_max = input_ids.shape
#     attn_mask = torch.zeros(B, L_max, dtype=torch.bool)
#     for i, L_i in enumerate(lengths):
#         attn_mask[i, :L_i] = True

#     expr = torch.stack(expr_list, dim=0)  # (B, num_genes)

#     return {
#         "input_ids": input_ids,
#         "attn_mask": attn_mask,
#         "expr": expr,
#     }

def spatial_collate_fn(batch):
    """
    batch: list of (ranked_gene_ids, expr, label)
      ranked_gene_ids: LongTensor [L_i]
      expr: FloatTensor [num_genes]
      label: scalar long
    """
    ranked_list, expr_list, label_list = zip(*batch)

    # pad variable-length sequences
    lengths = [x.size(0) for x in ranked_list]
    input_ids = pad_sequence(ranked_list, batch_first=True, padding_value=0)  # (B, L_max)

    B, L_max = input_ids.shape
    attn_mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, L_i in enumerate(lengths):
        attn_mask[i, :L_i] = True

    expr = torch.stack(expr_list, dim=0)              # (B, num_genes)
    labels = torch.tensor(label_list, dtype=torch.long)  # (B,)

    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "expr": expr,
        "label": labels,
    }