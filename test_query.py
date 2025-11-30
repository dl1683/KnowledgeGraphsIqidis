"""
Quick test for NL queries on existing graph.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph import KnowledgeGraph
from src.config import GEMINI_API_KEY

def main():
    print("="*60)
    print("NL QUERY TEST")
    print("="*60)

    kg = KnowledgeGraph("citiom_v_gulfstream_test", api_key=GEMINI_API_KEY)

    # Check stats first
    stats = kg.get_stats()
    print(f"\nGraph has {stats['entities']} entities, {stats['edges']} edges")

    if stats['entities'] == 0:
        print("No entities in graph. Run extraction first.")
        kg.close()
        return

    # Test queries
    test_queries = [
        "Who are the main parties in this case?",
        "What are the key allegations?",
        "Tell me about Gulfstream",
        "What obligations are mentioned?",
        "Who is involved in this dispute?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)

        result = kg.query(query)

        print(f"\nInterpretation: {result.interpretation}")
        print(f"Entities found: {len(result.entities)}")
        print(f"Edges found: {len(result.edges)}")
        print(f"Facts found: {len(result.facts)}")
        print(f"\nAnswer:\n{result.answer}")

    kg.close()
    print("\nQuery test complete!")


if __name__ == "__main__":
    main()
