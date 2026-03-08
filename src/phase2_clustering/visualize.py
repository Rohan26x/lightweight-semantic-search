# src/phase2_clustering/visualize.py
"""
Visualizations for cluster analysis.
Saves plots to models/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import MODELS_DIR


def plot_k_selection(k_sweep_results: dict, chosen_k: int) -> None:
    """FPC and inertia curves — justifies chosen K visually."""
    ks       = sorted(k_sweep_results.keys())
    fpcs     = [k_sweep_results[k]["fpc"]     for k in ks]
    inertias = [k_sweep_results[k]["inertia"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, fpcs, "o-", color="steelblue", linewidth=2)
    ax1.axvline(chosen_k, color="red", linestyle="--", label=f"Chosen K={chosen_k}")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Fuzzy Partition Coefficient (FPC)")
    ax1.set_title("FPC vs K  (higher = better separation)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ks, inertias, "o-", color="darkorange", linewidth=2)
    ax2.axvline(chosen_k, color="red", linestyle="--", label=f"Chosen K={chosen_k}")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("FCM Inertia (lower = tighter clusters)")
    ax2.set_title("Inertia vs K  (elbow method)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = MODELS_DIR / "k_selection.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out}")


def plot_umap_clusters(
    coords_2d: np.ndarray,
    memberships: np.ndarray,
    docs: list[dict],
) -> None:
    """UMAP scatter coloured by dominant cluster."""
    dominant = np.argmax(memberships, axis=1)
    K = memberships.shape[1]

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=dominant, cmap="tab20",
        s=3, alpha=0.6,
    )
    plt.colorbar(scatter, ax=ax, label="Dominant Cluster")
    ax.set_title(f"UMAP — {K} Fuzzy Clusters (coloured by dominant)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)

    out = MODELS_DIR / "umap_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out}")


def plot_umap_categories(
    coords_2d: np.ndarray,
    docs: list[dict],
) -> None:
    """UMAP scatter coloured by ground-truth category."""
    categories = [d["category"] for d in docs]
    unique_cats = sorted(set(categories))
    cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}
    colors      = [cat_to_idx[c] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=colors, cmap="tab20",
        s=3, alpha=0.5,
    )
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.cm.tab20(i / len(unique_cats)),
                   markersize=6, label=cat)
        for i, cat in enumerate(unique_cats)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=7)
    ax.set_title("UMAP — Ground-Truth Categories")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    out = MODELS_DIR / "umap_categories.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out}")


def plot_membership_heatmap(
    memberships: np.ndarray,
    n_sample: int = 500,
) -> None:
    """
    Heatmap of membership matrix for a random sample of documents.
    Shows the soft/fuzzy nature — most docs have spread across clusters.
    """
    idx = np.random.choice(len(memberships), size=min(n_sample, len(memberships)),
                           replace=False)
    sample = memberships[idx]

    # Sort rows by dominant cluster for readability
    order = np.argsort(np.argmax(sample, axis=1))
    sample = sample[order]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(sample, ax=ax, cmap="YlOrRd",
                xticklabels=range(memberships.shape[1]),
                yticklabels=False,
                cbar_kws={"label": "Membership"})
    ax.set_title(f"Fuzzy Membership Heatmap (sample={n_sample} docs)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Documents (sorted by dominant cluster)")

    out = MODELS_DIR / "membership_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out}")