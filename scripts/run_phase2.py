# scripts/run_phase2.py
"""
Phase 2 runner — Fuzzy Clustering end-to-end:
  1. Load embeddings from disk
  2. UMAP reduction: 384D → 30D (clustering) + 2D (visualization)
  3. K-sweep: test K=10..30 to justify cluster count
  4. Run final FCM with chosen K
  5. Save memberships + metadata + visualizations

Run from repo root:
    python scripts/run_phase2.py
"""

import json
import numpy as np
import time

from src.phase2_clustering.fuzzy_cluster import (
    reduce_dimensions, reduce_to_2d,
    find_best_k, run_fcm,
    save_memberships,
)
from src.phase2_clustering.cluster_eval import (
    load_cleaned_docs, build_cluster_metadata,
    cluster_category_distribution, boundary_documents,
)
from src.phase2_clustering.visualize import (
    plot_k_selection, plot_umap_clusters,
    plot_umap_categories, plot_membership_heatmap,
)
from src.phase1_ingest.embedder import load_embeddings
from src.config import MODELS_DIR


def main():
    t0 = time.time()

    # ── 1. Load embeddings ────────────────────────────────────────────────────
    print("\n── Step 1: Loading embeddings ──")
    embeddings, doc_ids = load_embeddings()
    docs = load_cleaned_docs()
    print(f"[Runner] Embeddings: {embeddings.shape}, docs: {len(docs)}")

    # Sanity check alignment
    assert len(embeddings) == len(docs), \
        f"Mismatch: {len(embeddings)} embeddings vs {len(docs)} docs"

    # ── 2. UMAP reduction ─────────────────────────────────────────────────────
    print("\n── Step 2: UMAP dimensionality reduction ──")
    reduced_30d = reduce_dimensions(embeddings, n_components=30)
    reduced_2d  = reduce_to_2d(embeddings)

    # Save 2D coords for later re-use in visualization
    np.save(MODELS_DIR / "umap_2d.npy", reduced_2d)
    print(f"[Runner] Saved 2D UMAP coords → {MODELS_DIR / 'umap_2d.npy'}")

    # ── 3. K-sweep to justify cluster count ───────────────────────────────────
    print("\n── Step 3: K-sweep (K=10 to 30) ──")
    k_sweep = find_best_k(reduced_30d, k_range=range(10, 31, 2))

    # Auto-select K: highest FPC with reasonable inertia
    # FPC penalises naturally as K grows, so we weight FPC heavily
    best_k = max(k_sweep, key=lambda k: k_sweep[k]["fpc"])
    print(f"\n[Runner] Auto-selected K={best_k} (highest FPC={k_sweep[best_k]['fpc']:.4f})")
    print("[Runner] Review models/k_selection.png to validate or override.")

    # ── 4. Final FCM run ──────────────────────────────────────────────────────
    print(f"\n── Step 4: Final Fuzzy C-Means with K={best_k} ──")
    memberships, fpc, inertia = run_fcm(reduced_30d, n_clusters=best_k)
    print(f"[Runner] Final FPC={fpc:.4f}, inertia={inertia:.2f}")

    # ── 5. Build metadata + save ──────────────────────────────────────────────
    print("\n── Step 5: Building cluster metadata ──")
    metadata = build_cluster_metadata(memberships, docs, k_sweep, best_k)
    save_memberships(memberships, metadata)

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    print("\n── Step 6: Generating visualizations ──")
    plot_k_selection(k_sweep, best_k)
    plot_umap_clusters(reduced_2d, memberships, docs)
    plot_umap_categories(reduced_2d, docs)
    plot_membership_heatmap(memberships)

    # ── 7. Print cluster summary ──────────────────────────────────────────────
    print("\n── Cluster Summary ──")
    for k in range(best_k):
        profile = metadata["cluster_profiles"][str(k)]
        top_cat = profile["top_categories"][0] if profile["top_categories"] else {}
        print(f"  Cluster {k:2d} | size={profile['size']:4d} | "
              f"top={top_cat.get('category','?')} ({top_cat.get('pct','?')}%)")

    print("\n── Top 5 Boundary Documents ──")
    for bd in metadata["boundary_documents"][:5]:
        print(f"  [{bd['category']}] entropy={bd['entropy']:.3f} | "
              f"clusters={bd['top2_clusters']} | "
              f"memberships={[round(m,3) for m in bd['top2_memberships']]}")
        print(f"  snippet: {bd['snippet'][:120]}...\n")

    elapsed = time.time() - t0
    print(f"\n✅ Phase 2 complete in {elapsed/60:.1f} minutes.")
    print(f"   Clusters         : {best_k}")
    print(f"   FPC              : {fpc:.4f}")
    print(f"   Memberships shape: {memberships.shape}")
    print(f"   Plots saved to   : {MODELS_DIR}/")


if __name__ == "__main__":
    main()