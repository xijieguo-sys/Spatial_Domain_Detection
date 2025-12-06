import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scanpy as sc
import numpy as np
import os

def plot_separate_loss_curves(total_losses, recon_losses, contra_pos_losses, contra_neg_losses, plot_dir):
    """Helper function to plot and save separate loss curves."""
    plt.figure(figsize=(10, 8))

    # Plot classification loss
    plt.plot(range(1, len(total_losses) + 1), total_losses, label='Total Loss', color='blue')

    # Plot domain loss
    plt.plot(range(1, len(recon_losses) + 1), recon_losses, label='Reconstruction Loss', color='orange')

   # Plot MMD loss
    plt.plot(range(1, len(contra_pos_losses) + 1), contra_pos_losses, label='Contrastive Positive Loss', color='green')

    # Plot total loss
    plt.plot(range(1, len(contra_neg_losses) + 1), contra_neg_losses, label='Contrastive Negative Loss', color='red')

    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "Loss_Curves.png"))
    plt.close()

def plot_ari_clustering(adata, true_color_indices, pred_color_indices, all_categories, ari, savedir):

    coords = adata.obsm["spatial"]
    colors = plt.cm.get_cmap("tab20", len(all_categories))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    scatter1 = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=true_color_indices, cmap=colors, s=20
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Ground Truth")

    scatter2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=pred_color_indices, cmap=colors, s=20
    )
    axes[1].invert_yaxis()
    axes[1].set_title(f"Predicted Clustering\n(ARI = {ari:.3f})")

    # legend_elements = [
    #     Patch(facecolor=colors(i), label=str(cat)) for i, cat in enumerate(all_categories)
    # ]
    num_clusters = len(np.unique(true_color_indices.tolist() + pred_color_indices.tolist()))
    legend_elements = [
        Patch(facecolor=colors(i), label=str(i)) for i in range(num_clusters)
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.18, 0.5), title="Layers / Clusters")

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    plot_dir = os.path.join(savedir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(os.path.join(savedir, 'plot', 'ari.png'), dpi=300, bbox_inches='tight')
    plt.close()
