"""
Generate visualization data from the knowledge graph.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.graph_exporter import GraphExporter


def main():
    # Configuration
    matter_name = "citiom_v_gulfstream"
    db_path = Path(__file__).parent / "matters" / matter_name / "graph.db"
    output_path = Path(__file__).parent / "visualization" / "graph_data.json"

    print(f"Loading graph from: {db_path}")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    exporter = GraphExporter(str(db_path))

    # Get stats first
    stats = exporter.get_stats()
    print(f"\nGraph Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Documents processed: {stats['total_documents']}")
    print(f"\n  Entities by type:")
    for etype, count in stats['entities_by_type'].items():
        print(f"    {etype}: {count}")

    # Export graph data (excluding Facts to keep it manageable)
    print(f"\nExporting graph data (excluding Facts)...")
    graph_data = exporter.get_graph_data(
        exclude_types=['Fact'],  # Exclude facts as there are too many
        min_connections=1,       # Only nodes with at least 1 connection
        limit_nodes=500          # Limit for performance
    )

    # Add stats to graph data
    graph_data['stats']['total_documents'] = stats['total_documents']
    graph_data['stats']['full_entity_counts'] = stats['entities_by_type']

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)

    print(f"\nExported to: {output_path}")
    print(f"  Nodes: {graph_data['stats']['total_nodes']}")
    print(f"  Links: {graph_data['stats']['total_links']}")
    print(f"  Entity types: {graph_data['stats']['entity_types']}")

    exporter.close()

    print(f"\n" + "="*50)
    print("To view the visualization:")
    print(f"  1. cd {output_path.parent}")
    print(f"  2. python -m http.server 8000")
    print(f"  3. Open http://localhost:8000 in browser")
    print("="*50)


if __name__ == "__main__":
    main()
