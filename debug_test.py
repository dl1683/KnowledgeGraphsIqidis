"""
Debug test to verify graph data and query functionality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph import KnowledgeGraph
from src.config import GEMINI_API_KEY

def main():
    print("="*60)
    print("DEBUG TEST")
    print("="*60)

    kg = KnowledgeGraph("citiom_v_gulfstream_test", api_key=GEMINI_API_KEY)

    # Get stats
    stats = kg.get_stats()
    print(f"\nGraph Statistics: {stats}")

    # List organizations
    print("\n" + "="*60)
    print("ORGANIZATIONS:")
    print("="*60)
    orgs = kg.list_entities("Organization", limit=20)
    for o in orgs:
        print(f"  - {o['name']} (confidence: {o['confidence']})")

    # List persons
    print("\n" + "="*60)
    print("PERSONS:")
    print("="*60)
    persons = kg.list_entities("Person", limit=20)
    for p in persons:
        print(f"  - {p['name']} (confidence: {p['confidence']})")

    # Search for specific entities
    print("\n" + "="*60)
    print("SEARCH: 'Gulfstream'")
    print("="*60)
    results = kg.search_entities("Gulfstream", limit=10)
    for r in results:
        print(f"  - {r['name']} ({r['type']})")

    print("\n" + "="*60)
    print("SEARCH: 'CITIOM'")
    print("="*60)
    results = kg.search_entities("CITIOM", limit=10)
    for r in results:
        print(f"  - {r['name']} ({r['type']})")

    # Get entity summary for main parties
    print("\n" + "="*60)
    print("ENTITY SUMMARY: Gulfstream")
    print("="*60)
    summary = kg.get_entity_summary("Gulfstream")
    print(summary)

    print("\n" + "="*60)
    print("ENTITY SUMMARY: CITIOM")
    print("="*60)
    summary = kg.get_entity_summary("CITIOM")
    print(summary)

    # Sample Facts
    print("\n" + "="*60)
    print("SAMPLE FACTS:")
    print("="*60)
    facts = kg.list_entities("Fact", limit=10)
    for f in facts:
        print(f"  - {f['name'][:80]}...")

    kg.close()
    print("\nDebug test complete!")


if __name__ == "__main__":
    main()
