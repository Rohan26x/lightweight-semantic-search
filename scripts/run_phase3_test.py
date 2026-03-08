# scripts/run_phase3_test.py
"""
Phase 3 manual test — demonstrates cache behaviour:
  1. Sends a query (miss)
  2. Sends the exact same query (hit)
  3. Sends a paraphrase (should hit at θ=0.85)
  4. Sends an unrelated query (should miss)
  5. Runs threshold analysis to show θ's effect on system behaviour

Run:
    python scripts/run_phase3_test.py
"""

import json
from src.phase3_cache.query_engine import QueryEngine


def separator(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def main():
    engine = QueryEngine(cache_threshold=0.85)

    test_queries = [
        # (label, query_text)
        ("FIRST QUERY  (expect MISS)", "What are the best graphics cards for gaming?"),
        ("EXACT REPEAT (expect HIT) ", "What are the best graphics cards for gaming?"),
        ("PARAPHRASE   (expect HIT) ", "Which GPU should I buy for playing video games?"),
        ("UNRELATED    (expect MISS)", "What is the best treatment for back pain?"),
        ("UNRELATED 2  (expect MISS)", "How do I configure a DNS server on Linux?"),
        ("PARAPHRASE 2 (expect HIT) ", "How to set up DNS on a Linux machine?"),
    ]

    separator("Cache Behaviour Test")
    for label, query in test_queries:
        result = engine.query(query)
        hit_str = "✅ HIT " if result["cache_hit"] else "❌ MISS"
        print(f"\n[{hit_str}] {label}")
        print(f"  Query   : {query}")
        if result["cache_hit"]:
            print(f"  Matched : {result['matched_query']}")
            print(f"  Score   : {result['similarity_score']}")
        print(f"  Cluster : {result['dominant_cluster']}")

    separator("Cache Stats")
    stats = engine.cache_stats()
    print(json.dumps(stats, indent=2))

    # ── Threshold analysis ────────────────────────────────────────────────────
    separator("Threshold Analysis — what each θ value reveals")
    probe_query = "Which video card is best for 3D rendering?"
    probe_emb   = engine.embed_query(probe_query)
    probe_cluster = engine.get_dominant_cluster(probe_emb)

    analysis = engine.cache.analyse_threshold(
        probe_emb, probe_cluster,
        thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    )

    print(f"\nProbe query: '{probe_query}'  (cluster={probe_cluster})")
    print(f"{'θ':>6}  {'Hit?':>5}  {'Similarity':>10}  Interpretation")
    print("-" * 80)
    for row in analysis:
        hit = "YES" if row["would_hit"] else "NO"
        sim = f"{row['similarity']:.4f}" if row["similarity"] else "N/A"
        print(f"  {row['threshold']:.2f}  {hit:>5}  {sim:>10}  {row['interpretation'][:65]}")

    separator("Done")
    print("\nKey insight: threshold θ controls the precision/recall tradeoff.")
    print("  High θ (0.95+) → high precision, low recall (misses paraphrases)")
    print("  Low  θ (0.70-) → high recall,    low precision (false hits)")
    print("  Sweet spot for paraphrase matching: θ ≈ 0.82–0.88")


if __name__ == "__main__":
    main()