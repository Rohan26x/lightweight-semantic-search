# src/phase3_cache/semantic_cache.py
"""
SemanticCache â€” hand-rolled semantic cache with cluster-scoped lookup.

Data structure:
    _store: dict[cluster_id â†’ list[CacheEntry]]

    CacheEntry holds:
        - query_text       : original query string
        - query_embedding  : np.ndarray (384,) â€” for similarity comparison
        - result           : the computed result to return on hit
        - dominant_cluster : int â€” which cluster bucket this lives in
        - timestamp        : float â€” for potential future TTL/LRU

Why a dict of lists (not a flat list)?
    Cluster-scoped lookup: on a new query we only scan the entries in its
    dominant cluster bucket, not the entire cache. This gives O(N/K) average
    lookup complexity instead of O(N). With K=10 and N=10000 cached entries,
    that's ~1000 comparisons instead of 10000.

Similarity metric: cosine similarity via dot product.
    Embeddings are L2-normalised at embed time (Phase 1 embedder), so
    cosine_sim(a, b) = dot(a, b). No sqrt needed â€” fast and exact.

The threshold Î¸ is the single most important tunable:
    - It defines what "close enough" means
    - Too high â†’ cache misses on paraphrases (defeats the purpose)
    - Too low  â†’ cache hits on unrelated queries (returns wrong results)
    - We expose exploration via the analyse_threshold() method
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from src.config import CACHE_SIMILARITY_THRESHOLD


@dataclass
class CacheEntry:
    query_text       : str
    query_embedding  : np.ndarray        # shape (D,), L2-normalised
    result           : str
    dominant_cluster : int
    timestamp        : float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-scoped semantic cache.

    Args:
        threshold : cosine similarity threshold Î¸ for cache hit (0.0â€“1.0)
                    Default loaded from config / .env
        n_clusters: number of cluster buckets (must match Phase 2 K)
    """

    def __init__(
        self,
        threshold : float = CACHE_SIMILARITY_THRESHOLD,
        n_clusters: int   = 10,
    ):
        self.threshold  = threshold
        self.n_clusters = n_clusters

        # Core data structure: cluster_id â†’ list of CacheEntry
        self._store: dict[int, list[CacheEntry]] = {
            k: [] for k in range(n_clusters)
        }

        # Stats
        self._hit_count  = 0
        self._miss_count = 0

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def lookup(
        self,
        query_embedding : np.ndarray,
        dominant_cluster: int,
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Search the cache for a semantically similar query.

        Only searches within the dominant_cluster bucket â€” this is the
        key efficiency decision. A query about hockey won't scan entries
        about politics.

        Args:
            query_embedding : L2-normalised embedding of the new query
            dominant_cluster: cluster index from Phase 2 membership vector

        Returns:
            (best_entry, similarity_score) if hit, else None
        """
        bucket = self._store.get(dominant_cluster, [])

        if not bucket:
            self._miss_count += 1
            return None

        # Vectorised cosine similarity against all entries in this bucket
        # Since embeddings are L2-normalised: cosine_sim = dot product
        stored_embeddings = np.stack([e.query_embedding for e in bucket])  # (M, D)
        similarities      = stored_embeddings @ query_embedding             # (M,)

        best_idx  = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            self._hit_count += 1
            return bucket[best_idx], best_score
        else:
            self._miss_count += 1
            return None

    def store(
        self,
        query_text      : str,
        query_embedding : np.ndarray,
        result          : str,
        dominant_cluster: int,
    ) -> CacheEntry:
        """
        Add a new entry to the appropriate cluster bucket.

        Called on every cache miss after the result is computed.
        """
        entry = CacheEntry(
            query_text       = query_text,
            query_embedding  = query_embedding,
            result           = result,
            dominant_cluster = dominant_cluster,
        )
        self._store[dominant_cluster].append(entry)
        return entry

    def flush(self) -> None:
        """Clear all cache entries and reset stats."""
        self._store      = {k: [] for k in range(self.n_clusters)}
        self._hit_count  = 0
        self._miss_count = 0

    # â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def total_entries(self) -> int:
        return sum(len(b) for b in self._store.values())

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return round(self._hit_count / total, 4) if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "hit_count"    : self._hit_count,
            "miss_count"   : self._miss_count,
            "hit_rate"     : self.hit_rate,
            "threshold"    : self.threshold,
            "bucket_sizes" : {str(k): len(v) for k, v in self._store.items()},
        }

    # â”€â”€ Threshold analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def analyse_threshold(
        self,
        query_embedding : np.ndarray,
        dominant_cluster: int,
        thresholds      : list[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    ) -> list[dict]:
        """
        For a given query, show what each threshold value would return.

        This is the exploration method the task asks for: it shows what
        each threshold value *reveals about system behaviour* rather than
        just picking a winner.

        Returns list of dicts, one per threshold, showing:
          - would_hit      : bool
          - best_match     : query text of the closest cached entry
          - similarity     : actual cosine similarity score
          - interpretation : human-readable explanation
        """
        bucket = self._store.get(dominant_cluster, [])
        if not bucket:
            return [{"threshold": t, "would_hit": False,
                     "best_match": None, "similarity": None,
                     "interpretation": "Empty bucket â€” miss regardless of threshold"}
                    for t in thresholds]

        stored_embeddings = np.stack([e.query_embedding for e in bucket])
        similarities      = stored_embeddings @ query_embedding
        best_idx          = int(np.argmax(similarities))
        best_score        = float(similarities[best_idx])
        best_entry        = bucket[best_idx]

        results = []
        for t in thresholds:
            would_hit = best_score >= t
            if would_hit:
                interp = (f"HIT â€” returns cached result for '{best_entry.query_text}' "
                          f"(sim={best_score:.4f} â‰¥ Î¸={t}). "
                          f"{'Correct paraphrase match.' if best_score > 0.90 else 'Borderline â€” verify semantic equivalence.'}")
            else:
                interp = (f"MISS â€” closest entry sim={best_score:.4f} < Î¸={t}. "
                          f"{'Threshold too strict â€” paraphrases won t match.' if best_score > 0.80 else 'Correctly rejected â€” queries are semantically distinct.'}")
            results.append({
                "threshold"    : t,
                "would_hit"    : would_hit,
                "best_match"   : best_entry.query_text,
                "similarity"   : round(best_score, 4),
                "interpretation": interp,
            })
        return results

    def __repr__(self) -> str:
        return (f"SemanticCache(threshold={self.threshold}, "
                f"entries={self.total_entries}, "
                f"hit_rate={self.hit_rate:.2%})")



