# src/phase3_cache/query_engine.py
"""
QueryEngine — ties together embedding, cluster assignment, cache lookup,
and ChromaDB retrieval into a single callable used by the FastAPI layer.

Flow for every query:
  1. Embed query on GPU
  2. Compute dominant cluster from Phase 2 memberships
  3. Check semantic cache
     a. HIT  → return cached result immediately
     b. MISS → query ChromaDB, format result, store in cache, return
"""

import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DEVICE, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CLUSTER_MEMBERSHIPS, CLUSTER_METADATA,
)
from src.phase3_cache.semantic_cache import SemanticCache


class QueryEngine:
    """
    Stateful engine that holds model, vectordb, cluster data, and cache
    in memory for the lifetime of the FastAPI process.
    """

    def __init__(self, cache_threshold: float = 0.85):
        print("[QueryEngine] Initialising...")

        # ── Embedding model on GPU ────────────────────────────────────────────
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print(f"[QueryEngine] Embedding model on {DEVICE.upper()}")

        # ── ChromaDB ──────────────────────────────────────────────────────────
        self.chroma_client     = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.chroma_collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
        print(f"[QueryEngine] ChromaDB collection: "
              f"{self.chroma_collection.count():,} docs")

        # ── Cluster data ──────────────────────────────────────────────────────
        self.memberships = np.load(CLUSTER_MEMBERSHIPS)   # (N, K)
        with open(CLUSTER_METADATA) as f:
            self.cluster_metadata = json.load(f)
        self.n_clusters = self.memberships.shape[1]
        print(f"[QueryEngine] Loaded cluster memberships: {self.memberships.shape}")

        # ── Semantic cache ────────────────────────────────────────────────────
        self.cache = SemanticCache(
            threshold  = cache_threshold,
            n_clusters = self.n_clusters,
        )
        print(f"[QueryEngine] SemanticCache ready (θ={cache_threshold})")
        print("[QueryEngine] ✅ Ready.")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string on GPU. Returns (D,) normalised vector."""
        vec = self.model.encode(
            [query],
            convert_to_numpy   = True,
            normalize_embeddings = True,
            device             = DEVICE,
        )
        return vec[0]   # (D,)

    def get_dominant_cluster(self, query_embedding: np.ndarray) -> int:
        """
        Assign query to its dominant cluster.

        We compute cosine similarity between the query embedding and the
        centroid of each cluster (approximated as the mean of top-N member
        embeddings). Here we use a simpler proxy: find the most similar
        document in each cluster and pick the cluster with highest similarity.

        Practical shortcut: dot-product the query against all stored
        memberships' weighted average. Since we have the full membership
        matrix, we can compute soft cluster centroids on the fly.

        For runtime efficiency we use the membership-weighted mean approach:
            centroid_k = mean of embeddings weighted by membership[:,k]
        But at query time we don't have all embeddings in RAM, so we instead
        use ChromaDB to find the nearest neighbour and inherit its dominant
        cluster — fast O(log N) lookup.
        """
        results = self.chroma_collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = 5,
            include          = ["metadatas", "embeddings"],
        )

        # Get the doc_ids of the top-5 nearest neighbours
        # Use their cluster memberships to vote on dominant cluster
        # (This avoids loading 16k embeddings into RAM at query time)
        top_ids = results["ids"][0]

        # Map doc_ids back to row indices via a lookup we build once
        if not hasattr(self, "_doc_id_to_idx"):
            self._build_doc_id_index()

        cluster_votes = np.zeros(self.n_clusters)
        for doc_id in top_ids:
            idx = self._doc_id_to_idx.get(doc_id)
            if idx is not None:
                cluster_votes += self.memberships[idx]

        dominant = int(np.argmax(cluster_votes))
        return dominant

    def _build_doc_id_index(self) -> None:
        """Build doc_id → row index lookup (called once on first query)."""
        import json
        from src.config import DOC_IDS_PATH
        with open(DOC_IDS_PATH) as f:
            doc_ids = json.load(f)
        self._doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        print(f"[QueryEngine] Built doc_id index ({len(self._doc_id_to_idx):,} entries)")

    def retrieve(self, query_embedding: np.ndarray, n_results: int = 5) -> str:
        """
        Query ChromaDB for top-N semantically similar documents.
        Returns formatted result string.
        """
        results = self.chroma_collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = n_results,
            include          = ["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            similarity = round(1 - dist, 4)   # ChromaDB returns cosine distance
            formatted.append(
                f"[Result {i+1}] category={meta['category']} "
                f"similarity={similarity}\n{doc[:300]}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, query_text: str) -> dict:
        """
        Main entry point — handles the full cache-aware query pipeline.

        Returns dict matching the FastAPI /query response schema.
        """
        # 1. Embed
        query_emb = self.embed_query(query_text)

        # 2. Dominant cluster
        dominant_cluster = self.get_dominant_cluster(query_emb)

        # 3. Cache lookup
        hit = self.cache.lookup(query_emb, dominant_cluster)

        if hit is not None:
            entry, score = hit
            return {
                "query"           : query_text,
                "cache_hit"       : True,
                "matched_query"   : entry.query_text,
                "similarity_score": round(score, 4),
                "result"          : entry.result,
                "dominant_cluster": dominant_cluster,
            }

        # 4. Cache miss — retrieve from ChromaDB
        result = self.retrieve(query_emb)

        # 5. Store in cache
        self.cache.store(query_text, query_emb, result, dominant_cluster)

        return {
            "query"           : query_text,
            "cache_hit"       : False,
            "matched_query"   : None,
            "similarity_score": None,
            "result"          : result,
            "dominant_cluster": dominant_cluster,
        }

    def flush_cache(self) -> None:
        self.cache.flush()

    def cache_stats(self) -> dict:
        return self.cache.stats()