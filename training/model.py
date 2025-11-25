from typing import Callable, Any, Dict

import lightning as L
import torch
import logging
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch.nn.functional as F
import os
import numpy as np
from module import InstanceEncoder, MLP

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score



class SpatialDomainModel(L.LightningModule):
    """
    Encoder (InstanceEncoder) + linear decoder trained
    to reconstruct the *original* gene expression vector.
    """
    def __init__(
        self,
        args,
        num_genes: int,
        cell_dim: int = 256,
        trans_heads: int = 1,
        trans_layer: int = 2,
        trans_ffd: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.num_genes = num_genes
        self.lr = lr
        self.weight_decay = weight_decay

        self.encoder = InstanceEncoder(
            num_genes=num_genes,
            d_model=cell_dim,
            n_heads=trans_heads,
            num_layers=trans_layer,
            dim_feedforward=trans_ffd,
            dropout=dropout
        )
        # decoder: embedding -> expression vector
        self.decoder = MLP(input_dim=cell_dim, output_dim=args.hvg_num, hidden_dim=args.dec_dims, dropout=args.dropout)

        self._test_embeddings = []
        self._test_labels = []

        # public attributes after test
        self.test_embeddings_ = None        # np.ndarray (N_test, d_model)
        self.test_labels_ = None            # np.ndarray (N_test,)
        self.test_mclust_labels_ = None     # np.ndarray (N_test,)
        self.test_ari_ = None               # float
        ## validation
        self._val_embeddings = []
        self._val_labels = []

        self.val_embeddings_ = None
        self.val_labels_ = None
        self.val_mclust_labels_ = None
        self.val_ari_ = None


    def forward(self, input_ids, attn_mask):
        """
        Returns reconstructed expression: (B, num_genes)
        """
        cell_emb = self.encoder(input_ids, attn_mask)  # (B, d_model)
        recon = self.decoder(cell_emb)                 # (B, num_genes)
        return recon

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        expr_true = batch["expr"]          # (B, num_genes)

        expr_pred = self(input_ids, attn_mask)

        # reconstruction loss on original expression
        loss = F.mse_loss(expr_pred, expr_true)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        For downstream clustering: return encoder embeddings only.
        """
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        emb = self.encoder(input_ids, attn_mask)
        return emb
    
    def on_test_start(self):
        self._test_embeddings = []
        self._test_labels = []
        self.test_embeddings_ = None
        self.test_labels_ = None
        self.test_mclust_labels_ = None
        self.test_ari_ = None

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        labels = batch["label"]           # (B,)

        emb = self.encoder(input_ids, attn_mask)  # (B, d_model)

        self._test_embeddings.append(emb.detach().cpu())
        self._test_labels.append(labels.detach().cpu())
        return emb

    def on_test_epoch_end(self):
        if len(self._test_embeddings) == 0:
            return

        emb = torch.cat(self._test_embeddings, dim=0).numpy()  # (N, d_model)
        labels = torch.cat(self._test_labels, dim=0).numpy()   # (N,)

        self.test_embeddings_ = emb
        self.test_labels_ = labels

        # 1) PCA on embeddings
        pca = PCA(n_components=20)
        emb_pca = pca.fit_transform(emb)  # (N, n_pcs)

        # 2) Decide number of clusters from GT labels (GraphST-style)
        num_cluster = len(np.unique(labels))

        # 3) Run Mclust via rpy2 (GraphST-style)
        mclust_labels = self._mclust_np(
            emb_pca,
            num_cluster=num_cluster,
            modelNames='EEE',
        )
        self.test_mclust_labels_ = mclust_labels

        # 4) Compute ARI
        ari = adjusted_rand_score(labels, mclust_labels)
        self.test_ari_ = ari
        self.log("test_ARI", float(ari), prog_bar=True)

        labels_npy_path = os.path.join(
            self.args.result_specific_folder,
            f"mclust_labels.npy",
        )
        np.save(labels_npy_path, mclust_labels)


        # clear caches
        self._test_embeddings = []
        self._test_labels = []

        

    def on_validation_start(self):
        self._val_embeddings = []
        self._val_labels = []
        self.val_embeddings_ = None
        self.val_labels_ = None
        self.val_mclust_labels_ = None
        self.val_ari_ = None

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        labels = batch["label"]        # (B,)

        emb = self.encoder(input_ids, attn_mask)  # (B, d_model)

        self._val_embeddings.append(emb.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        # you can optionally also compute recon loss here if you want val_loss
        # but not necessary if you only care about ARI

    def on_validation_epoch_end(self):
        if len(self._val_embeddings) == 0:
            return

        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score

        emb = torch.cat(self._val_embeddings, dim=0).numpy()
        labels = torch.cat(self._val_labels, dim=0).numpy()

        self.val_embeddings_ = emb
        self.val_labels_ = labels

        # PCA
        pca = PCA(n_components=20)
        emb_pca = pca.fit_transform(emb)

        # number of clusters from ground truth
        num_cluster = len(np.unique(labels))

        # run mclust in R
        mclust_labels = self._mclust_np(
            emb_pca,
            num_cluster=num_cluster,
            modelNames='EEE'
        )
        self.val_mclust_labels_ = mclust_labels

        # ARI
        ari = adjusted_rand_score(labels, mclust_labels)
        self.val_ari_ = ari
        self.log("val_ARI", float(ari), prog_bar=True)

        # clear buffers
        self._val_embeddings = []
        self._val_labels = []

    def _mclust_np(self, X, num_cluster, modelNames="EEE"):
        """
        X: np.ndarray (N, d), PCA embeddings
        Returns: np.ndarray (N,) of cluster labels (int), mclust output.
        """
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri

        # np.random.seed(random_seed)
        numpy2ri.activate()

        robjects.r.library("mclust")
        r_set_seed = robjects.r["set.seed"]
        r_set_seed(self.args.other_random_state)
        rmclust = robjects.r["Mclust"]

        r_X = numpy2ri.numpy2rpy(X)
        res = rmclust(r_X, num_cluster, modelNames)
        # mclust result: classification is usually res["classification"], but your
        # original code used res[-2]; we keep that to match GraphST-style usage.
        mclust_res = np.array(res[-2])
        # mclust labels are 1..K; convert to 0..K-1 if you want
        mclust_res = mclust_res.astype(int)
        return mclust_res