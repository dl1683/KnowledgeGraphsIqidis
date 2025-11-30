"""
Quick test to verify rate limiting works with a single document.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph import KnowledgeGraph
from src.config import GEMINI_API_KEY

def main():
    print("="*60)
    print("QUICK EXTRACTION TEST (Single Document)")
    print("="*60)

    # Initialize knowledge graph
    kg = KnowledgeGraph("citiom_v_gulfstream_test", api_key=GEMINI_API_KEY)

    # Process just ONE document to test
    test_doc = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents\CITIOM Statement of Claim.pdf"

    print(f"\nProcessing: {test_doc}")
    doc_id = kg.add_document(test_doc)

    if doc_id:
        print(f"\nDocument processed successfully!")
        stats = kg.get_stats()
        print(f"\nGraph Statistics:")
        print(f"  Entities: {stats['entities']}")
        print(f"  Edges: {stats['edges']}")
        print(f"  Documents: {stats['documents']}")
        print(f"\nEntities by Type:")
        for t, c in stats['entities_by_type'].items():
            print(f"  {t}: {c}")

        # Show sample entities
        print(f"\nSample Persons:")
        for e in kg.list_entities("Person", limit=5):
            print(f"  - {e['name']}")

        print(f"\nSample Organizations:")
        for e in kg.list_entities("Organization", limit=5):
            print(f"  - {e['name']}")

        # Test a query
        print(f"\n{'='*60}")
        print("Testing NL Query...")
        print("="*60)
        result = kg.query("Who are the main parties in this case?")
        print(f"\nAnswer: {result.answer}")

    kg.close()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
