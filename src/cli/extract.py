#!/usr/bin/env python3
"""
CLI tool for batch document extraction.

Usage:
    python -m src.cli.extract --matter my_case
    python -m src.cli.extract --matter my_case --dir ./custom_docs
    python -m src.cli.extract --matter my_case --file ./document.pdf
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import KnowledgeGraph, MATTERS_DIR


def main():
    parser = argparse.ArgumentParser(
        description='Extract entities and relationships from legal documents'
    )
    parser.add_argument(
        '--matter', '-m',
        required=True,
        help='Matter name (creates a new matter directory if not exists)'
    )
    parser.add_argument(
        '--dir', '-d',
        help='Directory containing documents to process (default: matters/<matter>/documents)'
    )
    parser.add_argument(
        '--file', '-f',
        help='Single file to process'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively process subdirectories (default: True)'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (default: from environment GEMINI_API_KEY)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Knowledge Graph Document Extractor")
    print("=" * 60)
    print(f"Matter: {args.matter}")

    # Initialize knowledge graph
    kg_kwargs = {'matter_name': args.matter}
    if args.api_key:
        kg_kwargs['api_key'] = args.api_key

    try:
        kg = KnowledgeGraph(**kg_kwargs)
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        sys.exit(1)

    # Determine documents location
    if args.file:
        # Single file mode
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        print(f"Processing single file: {filepath}")
        start = time.time()
        doc_id = kg.add_document(str(filepath))
        elapsed = time.time() - start

        if doc_id:
            print(f"Successfully processed: {filepath.name} ({elapsed:.1f}s)")
        else:
            print(f"Failed or skipped: {filepath.name}")

    else:
        # Directory mode
        if args.dir:
            doc_dir = Path(args.dir)
        else:
            doc_dir = MATTERS_DIR / args.matter / "documents"

        if not doc_dir.exists():
            print(f"Error: Directory not found: {doc_dir}")
            print(f"Create the directory and add documents, or use --dir to specify a different path")
            sys.exit(1)

        print(f"Documents directory: {doc_dir}")
        print("-" * 60)

        # Find documents
        extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
        files = []
        for ext in extensions:
            if args.recursive:
                files.extend(doc_dir.rglob(ext))
            else:
                files.extend(doc_dir.glob(ext))

        files = sorted(set(files))
        print(f"Found {len(files)} documents")

        if not files:
            print("No documents found. Supported formats: PDF, DOCX, DOC, TXT")
            sys.exit(0)

        # Process documents
        success_count = 0
        skip_count = 0
        fail_count = 0
        total_start = time.time()

        for i, filepath in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {filepath.name}")
            start = time.time()

            try:
                doc_id = kg.add_document(str(filepath))
                elapsed = time.time() - start

                if doc_id:
                    success_count += 1
                    print(f"  Success ({elapsed:.1f}s)")
                else:
                    skip_count += 1
                    print(f"  Skipped (already processed or empty)")

            except Exception as e:
                fail_count += 1
                print(f"  Error: {e}")

            # Show ETA
            if i < len(files):
                avg_time = (time.time() - total_start) / i
                remaining = (len(files) - i) * avg_time
                print(f"  ETA: {remaining/60:.1f} minutes remaining")

        # Summary
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print("Extraction Complete")
        print("=" * 60)
        print(f"Processed: {success_count}")
        print(f"Skipped:   {skip_count}")
        print(f"Failed:    {fail_count}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")

        # Show graph stats
        stats = kg.get_stats()
        print("\nGraph Statistics:")
        print(f"  Entities: {stats.get('entities', 0)}")
        print(f"  Edges:    {stats.get('edges', 0)}")
        print(f"  Documents: {stats.get('documents', 0)}")

    kg.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
