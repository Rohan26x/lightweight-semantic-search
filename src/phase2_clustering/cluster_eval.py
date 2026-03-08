# src/phase2_clustering/cluster_eval.py
"""
Cluster evaluation and semantic analysis.

Answers the key questions a sceptical reader would ask:
  1. Are the clusters semantically meaningful?
  2. What documents live at cluster boundaries (high uncertainty)?
  3. Where is the model genuinely uncertain?
"""

import json
import numpy as np
from collections import Counter
from src.config import CLEANED_DOCS_PATH


def load_cleaned_docs() -> list[dict]:
    docs = []
    with open(CLEANED_DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def get_dominant_cluster(memberships: np.ndarray) -> np.ndarray:
    """Hard assignment for analysis: index of highest-membership cluster per doc."""
    return np.argmax(memberships, axis=1)


def get_cluster_entropy(memberships: np.ndarray) -> np.ndarray:
    """
    Per-document entropy of membership distribution.
    High entropy = uncertain / boundary document.
    Low entropy  = clearly belongs to one cluster.
    """
    eps = 1e-10
    return -np.sum(memberships * np.log(memberships + eps), axis=1)


def top_docs_per_cluster(
    memberships: np.ndarray,
    docs: list[dict],
    n_top: int = 5,
) -> dict:
    """
    For each cluster, find the n_top most representative documents
    (highest membership score for that cluster).
    """
    K = memberships.shape[1]
    result = {}

    for k in range(K):
        cluster_scores = memberships[:, k]
        top_indices = np.argsort(cluster_scores)[::-1][:n_top]
        result[k] = [
            {
                "doc_id"    : docs[i]["doc_id"],
                "category"  : docs[i]["category"],
                "membership": float(cluster_scores[i]),
                "snippet"   : docs[i]["text"][:200],
            }
            for i in top_indices
        ]
    return result


def boundary_documents(
    memberships: np.ndarray,
    docs: list[dict],
    n: int = 20,
) -> list[dict]:
    """
    Find documents with highest entropy — these sit at cluster boundaries
    and are the most semantically ambiguous. Most interesting for analysis.
    """
    entropy = get_cluster_entropy(memberships)
    top_idx = np.argsort(entropy)[::-1][:n]

    results = []
    for i in top_idx:
        top2 = np.argsort(memberships[i])[::-1][:2]
        results.append({
            "doc_id"          : docs[i]["doc_id"],
            "category"        : docs[i]["category"],
            "entropy"         : float(entropy[i]),
            "top2_clusters"   : [int(c) for c in top2],
            "top2_memberships": [float(memberships[i, c]) for c in top2],
            "snippet"         : docs[i]["text"][:300],
        })
    return results


def cluster_category_distribution(
    memberships: np.ndarray,
    docs: list[dict],
) -> dict:
    """
    For each cluster, show the distribution of ground-truth categories.
    This is the primary evidence that clusters are semantically meaningful:
    a good cluster should be dominated by 1-2 related categories.
    """
    dominant = get_dominant_cluster(memberships)
    K = memberships.shape[1]
    result = {}

    for k in range(K):
        cluster_doc_indices = np.where(dominant == k)[0]
        categories = [docs[i]["category"] for i in cluster_doc_indices]
        dist = Counter(categories)
        total = len(categories)
        result[k] = {
            "size": total,
            "top_categories": [
                {"category": cat, "count": cnt, "pct": round(cnt/total*100, 1)}
                for cat, cnt in dist.most_common(5)
            ]
        }
    return result


def build_cluster_metadata(
    memberships: np.ndarray,
    docs: list[dict],
    k_sweep_results: dict,
    chosen_k: int,
) -> dict:
    """Assemble full metadata dict to persist alongside memberships."""
    cat_dist   = cluster_category_distribution(memberships, docs)
    top_docs   = top_docs_per_cluster(memberships, docs, n_top=3)
    boundaries = boundary_documents(memberships, docs, n=10)

    return {
        "chosen_k"         : chosen_k,
        "k_sweep_results"  : {str(k): v for k, v in k_sweep_results.items()},
        "cluster_profiles" : {
            str(k): {
                "size"          : cat_dist[k]["size"],
                "top_categories": cat_dist[k]["top_categories"],
                "top_docs"      : top_docs[k],
            }
            for k in range(chosen_k)
        },
        "boundary_documents": boundaries,
    }