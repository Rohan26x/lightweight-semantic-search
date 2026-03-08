# src/phase2_clustering/fuzzy_cluster.py
"""
Fuzzy C-Means clustering on document embeddings.

Pipeline:
  1. Load embeddings (16294, 384)
  2. Reduce to lower dims with UMAP — FCM degrades in high dimensions
     (curse of dimensionality makes all distances similar in 384-D space)
  3. Run Fuzzy C-Means for a range of K values
  4. Select best K using FPC (Fuzzy Partition Coefficient) + silhouette
  5. Save soft membership matrix (N, K) and cluster metadata

Design decision — UMAP before clustering:
  - 384 dims → 30 dims preserves local structure while making distance
    metrics meaningful for clustering
  - UMAP is run on GPU via cuml if available, else CPU (still fast at 16k docs)

Design decision — Fuzzy C-Means (FCM):
  - skfuzzy.cmeans gives membership matrix U of shape (K, N)
  - Each column sums to 1.0 → true probability distribution per doc
  - m=2.0 is standard fuzziness parameter (m=1 → hard clustering, m→∞ → uniform)
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import skfuzzy as fuzz
import umap
from sklearn.preprocessing import normalize

from src.config import (
    EMBEDDINGS_PATH, DOC_IDS_PATH,
    CLUSTER_MEMBERSHIPS, CLUSTER_METADATA, MODELS_DIR,
    N_CLUSTERS, DEVICE,
)


# ── UMAP reduction ────────────────────────────────────────────────────────────

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    UMAP dimensionality reduction: (N, 384) → (N, n_components).

    n_components=30: enough to preserve global structure for clustering
    while removing the noise that hurts FCM in high dimensions.
    n_components=2 is only for visualization (Phase 2 viz step).

    n_neighbors=15: local neighborhood size. Lower = more local structure,
    higher = more global. 15 is standard for ~16k docs.
    """
    print(f"[UMAP] Reducing {embeddings.shape} → (N, {n_components})...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",        # consistent with how embeddings were normalized
        random_state=random_state,
        low_memory=False,       # faster, fine for 16k docs
        verbose=True,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"[UMAP] Reduction complete → {reduced.shape}")
    return reduced.astype(np.float32)


def reduce_to_2d(
    embeddings: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Separate 2D reduction purely for visualization."""
    print("[UMAP-2D] Reducing to 2D for visualization...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


# ── Fuzzy C-Means ─────────────────────────────────────────────────────────────

def run_fcm(
    data: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    error: float = 0.005,
    maxiter: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, float, float]:
    """
    Run Fuzzy C-Means on reduced embeddings.

    Args:
        data       : (N, D) float32 array — UMAP-reduced embeddings
        n_clusters : number of fuzzy clusters K
        m          : fuzziness exponent (2.0 is standard)
        error      : convergence threshold
        maxiter    : max iterations

    Returns:
        memberships : (N, K) array — soft cluster assignments, rows sum to 1
        fpc         : Fuzzy Partition Coefficient (0→1, higher=better separation)
        inertia     : sum of weighted distances (lower=better)

    skfuzzy.cmeans expects data as (D, N) — we transpose accordingly.
    """
    np.random.seed(seed)
    data_T = data.T  # skfuzzy expects (features, samples)

    print(f"[FCM] Running Fuzzy C-Means: K={n_clusters}, m={m}, "
          f"data={data.shape}...")

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data_T,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=seed,
    )

    memberships = u.T  # (N, K) — one row per document
    inertia = float(jm[-1])

    print(f"[FCM] Done — FPC={fpc:.4f}, final inertia={inertia:.2f}, "
          f"iterations={len(jm)}")

    return memberships, fpc, inertia


# ── K selection ───────────────────────────────────────────────────────────────

def find_best_k(
    data: np.ndarray,
    k_range: range = range(10, 31, 2),
    m: float = 2.0,
) -> dict:
    """
    Sweep K values and collect FPC + inertia to justify cluster count choice.

    FPC (Fuzzy Partition Coefficient):
      - Range [1/K, 1]. Higher = crisper, better-separated clusters.
      - We look for the 'elbow' where FPC stops improving significantly.

    Returns dict mapping K → {fpc, inertia} for plotting.
    """
    print(f"\n[K-Selection] Sweeping K={list(k_range)}...")
    results = {}

    for k in tqdm(k_range, desc="K sweep"):
        _, fpc, inertia = run_fcm(data, n_clusters=k, m=m)
        results[k] = {"fpc": fpc, "inertia": inertia}
        print(f"  K={k:2d} → FPC={fpc:.4f}, inertia={inertia:.1f}")

    return results


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_memberships(memberships: np.ndarray, metadata: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CLUSTER_MEMBERSHIPS, memberships)
    with open(CLUSTER_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[FCM] Saved memberships → {CLUSTER_MEMBERSHIPS}")
    print(f"[FCM] Saved metadata    → {CLUSTER_METADATA}")


def load_memberships() -> tuple[np.ndarray, dict]:
    memberships = np.load(CLUSTER_MEMBERSHIPS)
    with open(CLUSTER_METADATA) as f:
        metadata = json.load(f)
    return memberships, metadata