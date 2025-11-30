#!/usr/bin/env python3
"""
CLI tool for exporting knowledge graph data.

Usage:
    python -m src.cli.export --matter my_case --output graph.json
    python -m src.cli.export --matter my_case --format d3 --output viz.json
    python -m src.cli.export --matter my_case --format csv --output entities.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import KnowledgeGraph


def export_json(kg: KnowledgeGraph, output_path: str):
    """Export full graph to JSON."""
    kg.export_graph(output_path)
    print(f"Exported full graph to: {output_path}")


def export_d3(kg: KnowledgeGraph, output_path: str, max_nodes: int = 500):
    """Export graph in D3.js compatible format."""
    data = kg.get_visualization_data(max_nodes=max_nodes)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Exported D3 visualization data to: {output_path}")
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Links: {len(data['links'])}")


def export_csv_entities(kg: KnowledgeGraph, output_path: str):
    """Export entities to CSV."""
    entities = kg.list_entities(limit=10000)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'type', 'confidence'])
        writer.writeheader()
        writer.writerows(entities)

    print(f"Exported {len(entities)} entities to: {output_path}")


def export_csv_edges(kg: KnowledgeGraph, output_path: str):
    """Export edges to CSV."""
    # Get all edges via the database
    edges = kg.db.get_all_edges(limit=50000)

    rows = []
    for e in edges:
        source = kg.db.get_entity(e.source_entity_id)
        target = kg.db.get_entity(e.target_entity_id)
        rows.append({
            'id': e.id,
            'source_id': e.source_entity_id,
            'source_name': source.canonical_name if source else 'Unknown',
            'target_id': e.target_entity_id,
            'target_name': target.canonical_name if target else 'Unknown',
            'relation_type': e.relation_type,
            'confidence': e.confidence
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'source_id', 'source_name', 'target_id', 'target_name', 'relation_type', 'confidence'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} edges to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export knowledge graph data to various formats'
    )
    parser.add_argument(
        '--matter', '-m',
        required=True,
        help='Matter name'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file path'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'd3', 'csv-entities', 'csv-edges'],
        default='json',
        help='Export format (default: json)'
    )
    parser.add_argument(
        '--max-nodes',
        type=int,
        default=500,
        help='Maximum nodes for D3 export (default: 500)'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (default: from environment)'
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Knowledge Graph Exporter")
    print("=" * 50)
    print(f"Matter: {args.matter}")
    print(f"Format: {args.format}")

    # Initialize knowledge graph
    kg_kwargs = {'matter_name': args.matter}
    if args.api_key:
        kg_kwargs['api_key'] = args.api_key

    try:
        kg = KnowledgeGraph(**kg_kwargs)
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        sys.exit(1)

    # Show current stats
    stats = kg.get_stats()
    print(f"\nGraph contains:")
    print(f"  Entities:  {stats.get('entities', 0)}")
    print(f"  Edges:     {stats.get('edges', 0)}")
    print(f"  Documents: {stats.get('documents', 0)}")
    print()

    # Export based on format
    try:
        if args.format == 'json':
            export_json(kg, args.output)
        elif args.format == 'd3':
            export_d3(kg, args.output, args.max_nodes)
        elif args.format == 'csv-entities':
            export_csv_entities(kg, args.output)
        elif args.format == 'csv-edges':
            export_csv_edges(kg, args.output)

    except Exception as e:
        print(f"Export error: {e}")
        sys.exit(1)

    kg.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
