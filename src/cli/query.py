#!/usr/bin/env python3
"""
CLI tool for querying the knowledge graph.

Usage:
    python -m src.cli.query --matter my_case "Who are the main parties?"
    python -m src.cli.query --matter my_case --interactive
    python -m src.cli.query --matter my_case --list-entities Person
"""
import argparse
import sys
import io
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import KnowledgeGraph


def main():
    parser = argparse.ArgumentParser(
        description='Query the knowledge graph using natural language'
    )
    parser.add_argument(
        '--matter', '-m',
        required=True,
        help='Matter name'
    )
    parser.add_argument(
        'query',
        nargs='?',
        help='Natural language query'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - keep asking questions'
    )
    parser.add_argument(
        '--list-entities', '-l',
        metavar='TYPE',
        help='List entities of a specific type (e.g., Person, Organization)'
    )
    parser.add_argument(
        '--search', '-s',
        metavar='NAME',
        help='Search for entities by name'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show graph statistics'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (default: from environment)'
    )

    args = parser.parse_args()

    # Initialize knowledge graph
    kg_kwargs = {'matter_name': args.matter}
    if args.api_key:
        kg_kwargs['api_key'] = args.api_key

    try:
        kg = KnowledgeGraph(**kg_kwargs)
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        sys.exit(1)

    # Handle different modes
    if args.stats:
        stats = kg.get_stats()
        print("=" * 50)
        print(f"Knowledge Graph: {args.matter}")
        print("=" * 50)
        print(f"Entities:  {stats.get('entities', 0)}")
        print(f"Edges:     {stats.get('edges', 0)}")
        print(f"Documents: {stats.get('documents', 0)}")
        print(f"Mentions:  {stats.get('mentions', 0)}")

    elif args.list_entities:
        entities = kg.list_entities(entity_type=args.list_entities, limit=100)
        print(f"\n{args.list_entities} entities ({len(entities)} found):")
        print("-" * 50)
        for e in entities:
            print(f"  - {e['name']} ({e['confidence']})")

    elif args.search:
        results = kg.search_entities(args.search, limit=20)
        print(f"\nSearch results for '{args.search}' ({len(results)} found):")
        print("-" * 50)
        for e in results:
            print(f"  - [{e['type']}] {e['name']}")

    elif args.interactive:
        print("=" * 50)
        print("Knowledge Graph Query Interface")
        print(f"Matter: {args.matter}")
        print("=" * 50)
        print("Type your questions (or 'quit' to exit)")
        print()

        while True:
            try:
                query = input("Query> ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                result = kg.query(query)
                print("\n" + "-" * 50)
                print("Answer:")
                print(result.answer)
                print("-" * 50)
                print(f"Found {len(result.entities)} entities, {len(result.edges)} relationships")
                print()

            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()

    elif args.query:
        print("=" * 50)
        print(f"Query: {args.query}")
        print("=" * 50)

        result = kg.query(args.query)

        print("\nAnswer:")
        print("-" * 50)
        print(result.answer)
        print("-" * 50)

        if result.entities:
            print(f"\nEntities found ({len(result.entities)}):")
            for e in result.entities[:10]:
                print(f"  - [{e.get('type', '?')}] {e.get('canonical_name', 'Unknown')}")
            if len(result.entities) > 10:
                print(f"  ... and {len(result.entities) - 10} more")

        if result.facts:
            print(f"\nFacts found ({len(result.facts)}):")
            for f in result.facts[:5]:
                print(f"  - {f.get('text', '')[:100]}...")

    else:
        parser.print_help()

    kg.close()


if __name__ == '__main__':
    main()
