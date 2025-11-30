"""
Export knowledge graph data for visualization.
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


class GraphExporter:
    """Export graph data for web visualization."""

    # Color palette for entity types
    TYPE_COLORS = {
        'Organization': '#4285f4',  # Blue
        'Person': '#ea4335',        # Red
        'Document': '#fbbc04',      # Yellow
        'Date': '#34a853',          # Green
        'Money': '#ff6d01',         # Orange
        'Location': '#46bdc6',      # Teal
        'Reference': '#9c27b0',     # Purple
        'Fact': '#607d8b',          # Gray
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def get_graph_data(
        self,
        entity_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        min_connections: int = 0,
        limit_nodes: int = 500,
        include_facts: bool = False
    ) -> Dict:
        """
        Export graph data in D3.js compatible format.

        Args:
            entity_types: Only include these entity types
            exclude_types: Exclude these entity types
            min_connections: Only include nodes with at least this many connections
            limit_nodes: Maximum number of nodes to include
            include_facts: Whether to include Fact entities (can be many)

        Returns:
            Dict with 'nodes' and 'links' for D3.js
        """
        cursor = self.conn.cursor()

        # Build entity filter
        type_filter = ""
        if entity_types:
            types_str = ",".join(f"'{t}'" for t in entity_types)
            type_filter = f"AND type IN ({types_str})"
        elif exclude_types:
            types_str = ",".join(f"'{t}'" for t in exclude_types)
            type_filter = f"AND type NOT IN ({types_str})"
        elif not include_facts:
            type_filter = "AND type != 'Fact'"

        # Get entities with connection counts (optimized with pre-computed counts)
        cursor.execute(f'''
            WITH edge_counts AS (
                SELECT entity_id, SUM(cnt) as connections FROM (
                    SELECT source_entity_id as entity_id, COUNT(*) as cnt FROM edges GROUP BY source_entity_id
                    UNION ALL
                    SELECT target_entity_id as entity_id, COUNT(*) as cnt FROM edges GROUP BY target_entity_id
                ) GROUP BY entity_id
            )
            SELECT e.id, e.canonical_name, e.type, e.properties, e.confidence,
                   COALESCE(ec.connections, 0) as connections
            FROM entities e
            LEFT JOIN edge_counts ec ON e.id = ec.entity_id
            WHERE e.status = 'active'
            {type_filter}
            AND COALESCE(ec.connections, 0) >= ?
            ORDER BY connections DESC
            LIMIT ?
        ''', (min_connections, limit_nodes))

        entities = cursor.fetchall()
        entity_ids = {e['id'] for e in entities}

        # Build nodes
        nodes = []
        id_to_index = {}
        for i, e in enumerate(entities):
            id_to_index[e['id']] = i
            nodes.append({
                'id': i,
                'entity_id': e['id'],
                'name': e['canonical_name'][:50],  # Truncate long names
                'full_name': e['canonical_name'],
                'type': e['type'],
                'color': self.TYPE_COLORS.get(e['type'], '#999999'),
                'connections': e['connections'],
                'confidence': e['confidence'],
                'properties': json.loads(e['properties']) if e['properties'] else {}
            })

        # Get edges between included entities
        cursor.execute('''
            SELECT source_entity_id, target_entity_id, relation_type, confidence, properties
            FROM edges
            WHERE source_entity_id IN ({}) AND target_entity_id IN ({})
        '''.format(
            ','.join(f"'{eid}'" for eid in entity_ids),
            ','.join(f"'{eid}'" for eid in entity_ids)
        ))

        edges = cursor.fetchall()

        # Build links (deduplicate)
        seen_links = set()
        links = []
        for e in edges:
            source_idx = id_to_index.get(e['source_entity_id'])
            target_idx = id_to_index.get(e['target_entity_id'])

            if source_idx is not None and target_idx is not None:
                link_key = (source_idx, target_idx, e['relation_type'])
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append({
                        'source': source_idx,
                        'target': target_idx,
                        'relation': e['relation_type'],
                        'confidence': e['confidence']
                    })

        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'entity_types': list(set(n['type'] for n in nodes))
            }
        }

    def get_entity_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        max_nodes: int = 100
    ) -> Dict:
        """
        Get neighborhood graph around a specific entity.

        Args:
            entity_id: The central entity
            depth: How many hops to include
            max_nodes: Maximum nodes to include

        Returns:
            Dict with 'nodes' and 'links' for D3.js
        """
        cursor = self.conn.cursor()

        # BFS to find neighbors
        visited = {entity_id}
        frontier = {entity_id}

        for _ in range(depth):
            if len(visited) >= max_nodes:
                break

            # Find neighbors of frontier
            frontier_str = ','.join(f"'{eid}'" for eid in frontier)
            cursor.execute(f'''
                SELECT DISTINCT
                    CASE WHEN source_entity_id IN ({frontier_str}) THEN target_entity_id ELSE source_entity_id END as neighbor
                FROM edges
                WHERE source_entity_id IN ({frontier_str}) OR target_entity_id IN ({frontier_str})
            ''')

            new_frontier = set()
            for row in cursor.fetchall():
                if row['neighbor'] not in visited and len(visited) < max_nodes:
                    visited.add(row['neighbor'])
                    new_frontier.add(row['neighbor'])

            frontier = new_frontier
            if not frontier:
                break

        # Get entity data
        entity_ids_str = ','.join(f"'{eid}'" for eid in visited)
        cursor.execute(f'''
            SELECT id, canonical_name, type, properties, confidence
            FROM entities
            WHERE id IN ({entity_ids_str})
        ''')

        entities = cursor.fetchall()
        id_to_index = {}
        nodes = []

        for i, e in enumerate(entities):
            id_to_index[e['id']] = i
            nodes.append({
                'id': i,
                'entity_id': e['id'],
                'name': e['canonical_name'][:50],
                'full_name': e['canonical_name'],
                'type': e['type'],
                'color': self.TYPE_COLORS.get(e['type'], '#999999'),
                'is_center': e['id'] == entity_id,
                'properties': json.loads(e['properties']) if e['properties'] else {}
            })

        # Get edges
        cursor.execute(f'''
            SELECT source_entity_id, target_entity_id, relation_type, confidence
            FROM edges
            WHERE source_entity_id IN ({entity_ids_str}) AND target_entity_id IN ({entity_ids_str})
        ''')

        links = []
        seen = set()
        for e in cursor.fetchall():
            source_idx = id_to_index.get(e['source_entity_id'])
            target_idx = id_to_index.get(e['target_entity_id'])
            if source_idx is not None and target_idx is not None:
                key = (source_idx, target_idx, e['relation_type'])
                if key not in seen:
                    seen.add(key)
                    links.append({
                        'source': source_idx,
                        'target': target_idx,
                        'relation': e['relation_type']
                    })

        return {'nodes': nodes, 'links': links}

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT type, COUNT(*) as count FROM entities GROUP BY type ORDER BY count DESC')
        entity_counts = {row['type']: row['count'] for row in cursor.fetchall()}

        cursor.execute('SELECT relation_type, COUNT(*) as count FROM edges GROUP BY relation_type ORDER BY count DESC')
        relation_counts = {row['relation_type']: row['count'] for row in cursor.fetchall()}

        cursor.execute('SELECT COUNT(*) FROM entities')
        total_entities = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM edges')
        total_edges = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM documents WHERE processed_at IS NOT NULL')
        total_docs = cursor.fetchone()[0]

        return {
            'total_entities': total_entities,
            'total_edges': total_edges,
            'total_documents': total_docs,
            'entities_by_type': entity_counts,
            'edges_by_type': relation_counts,
            'type_colors': self.TYPE_COLORS
        }

    def search_entities(self, query: str, limit: int = 20) -> List[Dict]:
        """Search entities by name."""
        cursor = self.conn.cursor()
        cursor.execute('''
            WITH edge_counts AS (
                SELECT entity_id, SUM(cnt) as connections FROM (
                    SELECT source_entity_id as entity_id, COUNT(*) as cnt FROM edges GROUP BY source_entity_id
                    UNION ALL
                    SELECT target_entity_id as entity_id, COUNT(*) as cnt FROM edges GROUP BY target_entity_id
                ) GROUP BY entity_id
            )
            SELECT e.id, e.canonical_name, e.type, COALESCE(ec.connections, 0) as connections
            FROM entities e
            LEFT JOIN edge_counts ec ON e.id = ec.entity_id
            WHERE e.canonical_name LIKE ?
            ORDER BY connections DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        return [
            {
                'id': row['id'],
                'name': row['canonical_name'],
                'type': row['type'],
                'connections': row['connections'],
                'color': self.TYPE_COLORS.get(row['type'], '#999999')
            }
            for row in cursor.fetchall()
        ]
