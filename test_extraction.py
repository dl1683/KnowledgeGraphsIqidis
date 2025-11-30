"""
Test script for the Knowledge Graph extraction pipeline.

Tests:
1. Document parsing
2. Structural extraction
3. Semantic extraction
4. Entity resolution
5. NL queries
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph import KnowledgeGraph
from src.config import GEMINI_API_KEY


def test_extraction_pipeline():
    """Test the full extraction pipeline on CITIOM v Gulfstream documents."""

    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH EXTRACTION TEST")
    print("="*80)

    # Initialize knowledge graph for this matter
    kg = KnowledgeGraph("citiom_v_gulfstream", api_key=GEMINI_API_KEY)

    # Document directory
    doc_dir = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"

    print(f"\nProcessing documents from: {doc_dir}")
    print("This will take some time...")

    # Process all documents
    doc_ids = kg.add_documents_from_directory(doc_dir, recursive=True)

    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(doc_ids)} documents")

    # Show statistics
    stats = kg.get_stats()
    print(f"\nGraph Statistics:")
    print(f"  Total Entities: {stats['entities']}")
    print(f"  Total Edges: {stats['edges']}")
    print(f"  Total Documents: {stats['documents']}")
    print(f"  Total Mentions: {stats['mentions']}")
    print(f"  Pending Resolutions: {stats['pending_resolutions']}")

    print(f"\nEntities by Type:")
    for entity_type, count in stats['entities_by_type'].items():
        print(f"  {entity_type}: {count}")

    # Show sample entities
    print(f"\n{'='*80}")
    print("SAMPLE ENTITIES")
    print("="*80)

    for entity_type in ['Person', 'Organization', 'Fact']:
        entities = kg.list_entities(entity_type, limit=5)
        if entities:
            print(f"\n{entity_type}s:")
            for e in entities:
                print(f"  - {e['name']} (confidence: {e['confidence']})")

    kg.close()
    return True


def test_nl_queries():
    """Test natural language queries on the knowledge graph."""

    print("\n" + "="*80)
    print("NATURAL LANGUAGE QUERY TEST")
    print("="*80)

    # Initialize knowledge graph
    kg = KnowledgeGraph("citiom_v_gulfstream", api_key=GEMINI_API_KEY)

    # Check if we have data
    stats = kg.get_stats()
    if stats['entities'] == 0:
        print("No entities in graph. Run extraction first.")
        kg.close()
        return False

    # Test queries
    test_queries = [
        "Who are the main parties in this case?",
        "What is the relationship between CITIOM and Gulfstream?",
        "What are the key allegations in this case?",
        "What obligations does Gulfstream have?",
        "Who are the witnesses in this case?",
        "What deadlines are mentioned in the documents?",
        "What is the case about?",
    ]

    for query in test_queries:
        print(f"\n{'-'*60}")
        result = kg.query(query)
        print(f"\nAnswer: {result.answer}")
        print(f"\nFound {len(result.entities)} entities, {len(result.edges)} relationships")

    kg.close()
    return True


def test_entity_summary():
    """Test entity summary functionality."""

    print("\n" + "="*80)
    print("ENTITY SUMMARY TEST")
    print("="*80)

    kg = KnowledgeGraph("citiom_v_gulfstream", api_key=GEMINI_API_KEY)

    # Find some entities to summarize
    entities = kg.list_entities(limit=5)

    for e in entities[:3]:
        print(f"\n{'-'*60}")
        summary = kg.get_entity_summary(e['name'])
        print(summary)

    kg.close()
    return True


def interactive_mode():
    """Interactive query mode."""

    print("\n" + "="*80)
    print("INTERACTIVE QUERY MODE")
    print("="*80)
    print("Type your questions (or 'quit' to exit)")

    kg = KnowledgeGraph("citiom_v_gulfstream", api_key=GEMINI_API_KEY)

    while True:
        print("\n" + "-"*40)
        query = input("Question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        if query.startswith('/'):
            # Command mode
            cmd = query[1:].lower().split()
            if cmd[0] == 'stats':
                stats = kg.get_stats()
                print(f"\nGraph Statistics: {stats}")
            elif cmd[0] == 'entities':
                entity_type = cmd[1] if len(cmd) > 1 else None
                entities = kg.list_entities(entity_type, limit=20)
                for e in entities:
                    print(f"  - {e['name']} ({e['type']})")
            elif cmd[0] == 'summary' and len(cmd) > 1:
                name = ' '.join(cmd[1:])
                print(kg.get_entity_summary(name))
            elif cmd[0] == 'help':
                print("Commands:")
                print("  /stats - Show graph statistics")
                print("  /entities [type] - List entities")
                print("  /summary <name> - Get entity summary")
                print("  /help - Show this help")
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            result = kg.query(query)
            print(f"\n{result.answer}")

    kg.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Graph Test Suite")
    parser.add_argument('--extract', action='store_true', help='Run extraction on documents')
    parser.add_argument('--query', action='store_true', help='Run query tests')
    parser.add_argument('--summary', action='store_true', help='Run entity summary tests')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive query mode')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests')

    args = parser.parse_args()

    if args.all or args.extract:
        test_extraction_pipeline()

    if args.all or args.query:
        test_nl_queries()

    if args.all or args.summary:
        test_entity_summary()

    if args.interactive:
        interactive_mode()

    if not any([args.extract, args.query, args.summary, args.interactive, args.all]):
        print("No test specified. Use --help for options.")
        print("\nRunning extraction pipeline by default...")
        test_extraction_pipeline()
