"""
REST API server for the Knowledge Graph system.

This module provides all the HTTP endpoints for CRUD operations on the knowledge graph.
It is decoupled from any specific frontend (visualization, CLI, etc.).

Usage:
    from src.api import create_app
    app = create_app("my_matter")
    app.run(host='0.0.0.0', port=5000)
"""
import json
import re
import uuid
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from flask import Flask, Blueprint, jsonify, request
from flask_cors import CORS

import google.generativeai as genai

# Import from core
from ..core import KnowledgeGraph, GEMINI_API_KEY, MATTERS_DIR
from ..core.storage.database import Database
from ..visualization.graph_exporter import GraphExporter


# Create API blueprint
api = Blueprint('api', __name__, url_prefix='/api')


# Global instances - initialized per matter
_kg: Optional[KnowledgeGraph] = None
_exporter: Optional[GraphExporter] = None
_matter_name: str = "default"
_nl_edit_model = None


def init_matter(matter_name: str, api_key: str = GEMINI_API_KEY):
    """Initialize the API for a specific matter."""
    global _kg, _exporter, _matter_name, _nl_edit_model
    _matter_name = matter_name
    _kg = KnowledgeGraph(matter_name, api_key=api_key)

    db_path = MATTERS_DIR / matter_name / "graph.db"
    _exporter = GraphExporter(str(db_path))

    # Configure NL edit model
    genai.configure(api_key=api_key)
    _nl_edit_model = genai.GenerativeModel('gemini-2.0-flash')


def get_kg() -> KnowledgeGraph:
    """Get the KnowledgeGraph instance."""
    if _kg is None:
        raise RuntimeError("API not initialized. Call init_matter() first.")
    return _kg


def get_exporter() -> GraphExporter:
    """Get the GraphExporter instance."""
    if _exporter is None:
        raise RuntimeError("API not initialized. Call init_matter() first.")
    return _exporter


# ==================== Statistics ====================

@api.route('/stats')
def api_stats():
    """Get graph statistics."""
    exp = get_exporter()
    stats = exp.get_stats()
    return jsonify(stats)


# ==================== Graph Data ====================

@api.route('/graph')
def api_graph():
    """Get full graph data for visualization."""
    exp = get_exporter()

    include_facts = request.args.get('include_facts', 'false').lower() == 'true'
    min_connections = int(request.args.get('min_connections', 0))
    limit = int(request.args.get('limit', 1000))

    exclude_types = None if include_facts else ['Fact']

    graph_data = exp.get_graph_data(
        exclude_types=exclude_types,
        min_connections=min_connections,
        limit_nodes=limit,
        include_facts=include_facts
    )

    stats = exp.get_stats()
    graph_data['stats']['total_documents'] = stats['total_documents']
    graph_data['stats']['full_entity_counts'] = stats['entities_by_type']
    graph_data['stats']['type_colors'] = stats['type_colors']

    return jsonify(graph_data)


# ==================== Search ====================

@api.route('/search')
def api_search():
    """Search entities by name."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))

    if len(query) < 2:
        return jsonify([])

    exp = get_exporter()
    results = exp.search_entities(query, limit)
    return jsonify(results)


# ==================== Entity CRUD ====================

@api.route('/entity/<entity_id>')
def api_get_entity(entity_id):
    """Get entity details and neighborhood."""
    depth = int(request.args.get('depth', 2))
    max_nodes = int(request.args.get('max_nodes', 50))

    exp = get_exporter()
    neighborhood = exp.get_entity_neighborhood(entity_id, depth, max_nodes)
    return jsonify(neighborhood)


@api.route('/entity/<entity_id>', methods=['PUT'])
def api_update_entity(entity_id):
    """Update entity properties."""
    data = request.get_json()

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        updates = []
        params = []

        if 'canonical_name' in data:
            updates.append('canonical_name = ?')
            params.append(data['canonical_name'])

        if 'type' in data:
            updates.append('type = ?')
            params.append(data['type'])

        if 'properties' in data:
            updates.append('properties = ?')
            params.append(json.dumps(data['properties']))

        if 'confidence' in data:
            updates.append('confidence = ?')
            params.append(data['confidence'])

        if updates:
            params.append(entity_id)
            cursor.execute(f'''
                UPDATE entities SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', params)
            exp.conn.commit()

        return jsonify({'success': True, 'entity_id': entity_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/entity/<entity_id>', methods=['DELETE'])
def api_delete_entity(entity_id):
    """Delete an entity and its relationships."""
    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        cursor.execute('DELETE FROM edges WHERE source_entity_id = ? OR target_entity_id = ?',
                      (entity_id, entity_id))
        cursor.execute('DELETE FROM mentions WHERE entity_id = ?', (entity_id,))
        cursor.execute('DELETE FROM aliases WHERE entity_id = ?', (entity_id,))
        cursor.execute('DELETE FROM entities WHERE id = ?', (entity_id,))

        exp.conn.commit()

        return jsonify({'success': True, 'deleted': entity_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/entity', methods=['POST'])
def api_create_entity():
    """Create a new entity."""
    data = request.get_json()

    if not data.get('canonical_name') or not data.get('type'):
        return jsonify({'error': 'canonical_name and type are required'}), 400

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        entity_id = str(uuid.uuid4())

        cursor.execute('''
            INSERT INTO entities (id, type, canonical_name, properties, confidence, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (
            entity_id,
            data['type'],
            data['canonical_name'],
            json.dumps(data.get('properties', {})),
            data.get('confidence', 'confirmed')
        ))

        exp.conn.commit()

        return jsonify({
            'success': True,
            'entity_id': entity_id,
            'entity': {
                'id': entity_id,
                'canonical_name': data['canonical_name'],
                'type': data['type']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Edge CRUD ====================

@api.route('/edge', methods=['POST'])
def api_create_edge():
    """Create a new edge/relationship."""
    data = request.get_json()

    required = ['source_entity_id', 'target_entity_id', 'relation_type']
    if not all(data.get(f) for f in required):
        return jsonify({'error': f'{required} are required'}), 400

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        edge_id = str(uuid.uuid4())

        cursor.execute('''
            INSERT INTO edges (id, source_entity_id, target_entity_id, relation_type, confidence, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            edge_id,
            data['source_entity_id'],
            data['target_entity_id'],
            data['relation_type'],
            data.get('confidence', 'confirmed'),
            json.dumps(data.get('properties', {}))
        ))

        exp.conn.commit()

        return jsonify({'success': True, 'edge_id': edge_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/edge/<edge_id>', methods=['DELETE'])
def api_delete_edge(edge_id):
    """Delete an edge."""
    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        cursor.execute('DELETE FROM edges WHERE id = ?', (edge_id,))
        exp.conn.commit()

        return jsonify({'success': True, 'deleted': edge_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Query ====================

@api.route('/query', methods=['POST'])
def api_query():
    """Execute NL query and return results with subgraph."""
    data = request.get_json()
    query_text = data.get('query', '')

    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    try:
        kg = get_kg()
        result = kg.query(query_text)
        exp = get_exporter()

        entity_ids = [e.get('id') for e in result.entities if e.get('id')]

        nodes = []
        id_to_index = {}

        for i, entity in enumerate(result.entities[:100]):
            entity_id = entity.get('id')
            if entity_id:
                id_to_index[entity_id] = i
                nodes.append({
                    'id': i,
                    'entity_id': entity_id,
                    'name': entity.get('canonical_name', 'Unknown')[:50],
                    'full_name': entity.get('canonical_name', 'Unknown'),
                    'type': entity.get('type', 'Unknown'),
                    'color': GraphExporter.TYPE_COLORS.get(entity.get('type'), '#999999'),
                    'properties': entity.get('properties', {})
                })

        links = []
        seen_links = set()
        for edge in result.edges[:200]:
            source_id = edge.get('source_entity_id')
            target_id = edge.get('target_entity_id')

            source_idx = id_to_index.get(source_id)
            target_idx = id_to_index.get(target_id)

            if source_idx is not None and target_idx is not None:
                link_key = (source_idx, target_idx, edge.get('relation_type', ''))
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append({
                        'source': source_idx,
                        'target': target_idx,
                        'relation': edge.get('relation_type', 'related_to')
                    })

        facts = []
        for fact in result.facts[:50]:
            facts.append({
                'id': fact.get('id'),
                'content': fact.get('canonical_name', ''),
                'type': fact.get('type', 'Fact')
            })

        return jsonify({
            'query': query_text,
            'answer': result.answer,
            'subgraph': {
                'nodes': nodes,
                'links': links
            },
            'facts': facts,
            'stats': {
                'entities_found': len(result.entities),
                'edges_found': len(result.edges),
                'facts_found': len(result.facts)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Merge ====================

@api.route('/merge', methods=['POST'])
def api_merge_entities():
    """Merge two entities into one."""
    data = request.get_json()

    keep_id = data.get('keep_id')
    merge_id = data.get('merge_id')

    if not keep_id or not merge_id:
        return jsonify({'error': 'keep_id and merge_id are required'}), 400

    try:
        kg = get_kg()
        kg.merge_entities(keep_id, merge_id)

        return jsonify({
            'success': True,
            'kept': keep_id,
            'merged': merge_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Entity/Relation Types ====================

@api.route('/entity-types')
def api_entity_types():
    """Get available entity types."""
    from ..core.config import ENTITY_TYPES
    return jsonify(ENTITY_TYPES)


@api.route('/relation-types')
def api_relation_types():
    """Get available relation types."""
    from ..core.config import RELATION_TYPES
    return jsonify(RELATION_TYPES)


# ==================== Natural Language Edit ====================

@api.route('/nl-edit', methods=['POST'])
def api_nl_edit():
    """Execute a natural language edit command."""
    global _nl_edit_model

    data = request.get_json()
    command = data.get('command', '').strip()

    if not command:
        return jsonify({'error': 'No command provided'}), 400

    try:
        prompt = f"""Parse this knowledge graph edit command and return a JSON object with the action.

Command: "{command}"

Return ONLY a JSON object (no markdown, no explanation) with one of these structures:

1. For rename: {{"action": "rename", "entity_name": "old name", "new_name": "new name"}}
2. For merge: {{"action": "merge", "entity_to_remove": "name to merge away", "entity_to_keep": "name to keep"}}
3. For delete: {{"action": "delete", "entity_name": "name to delete"}}
4. For create entity: {{"action": "create_entity", "name": "entity name", "type": "Person|Organization|Document|Date|Money|Location|Reference|Fact"}}
5. For create relationship: {{"action": "create_edge", "source": "source entity name", "target": "target entity name", "relation": "relationship type"}}
6. For change type: {{"action": "change_type", "entity_name": "entity name", "new_type": "new type"}}
7. If unclear: {{"action": "unknown", "message": "explanation of what's unclear"}}

JSON response:"""

        response = _nl_edit_model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)

        parsed = json.loads(response_text)
        action = parsed.get('action')

        exp = get_exporter()
        cursor = exp.conn.cursor()

        if action == 'unknown':
            return jsonify({
                'success': False,
                'error': parsed.get('message', 'Could not understand command'),
                'parsed': parsed
            })

        elif action == 'rename':
            entity_name = parsed.get('entity_name', '')
            new_name = parsed.get('new_name', '')

            cursor.execute('''
                SELECT id, canonical_name FROM entities
                WHERE canonical_name LIKE ?
                ORDER BY LENGTH(canonical_name)
                LIMIT 1
            ''', (f'%{entity_name}%',))
            row = cursor.fetchone()

            if not row:
                return jsonify({
                    'success': False,
                    'error': f'Entity "{entity_name}" not found'
                })

            cursor.execute('''
                UPDATE entities SET canonical_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_name, row['id']))
            exp.conn.commit()

            return jsonify({
                'success': True,
                'action': 'rename',
                'message': f'Renamed "{row["canonical_name"]}" to "{new_name}"',
                'entity_id': row['id']
            })

        elif action == 'merge':
            entity_to_remove = parsed.get('entity_to_remove', '')
            entity_to_keep = parsed.get('entity_to_keep', '')

            cursor.execute('SELECT id, canonical_name FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{entity_to_remove}%',))
            remove_row = cursor.fetchone()

            cursor.execute('SELECT id, canonical_name FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{entity_to_keep}%',))
            keep_row = cursor.fetchone()

            if not remove_row:
                return jsonify({'success': False, 'error': f'Entity "{entity_to_remove}" not found'})
            if not keep_row:
                return jsonify({'success': False, 'error': f'Entity "{entity_to_keep}" not found'})

            kg = get_kg()
            kg.merge_entities(keep_row['id'], remove_row['id'])

            return jsonify({
                'success': True,
                'action': 'merge',
                'message': f'Merged "{remove_row["canonical_name"]}" into "{keep_row["canonical_name"]}"'
            })

        elif action == 'delete':
            entity_name = parsed.get('entity_name', '')

            cursor.execute('SELECT id, canonical_name FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{entity_name}%',))
            row = cursor.fetchone()

            if not row:
                return jsonify({'success': False, 'error': f'Entity "{entity_name}" not found'})

            entity_id = row['id']
            cursor.execute('DELETE FROM edges WHERE source_entity_id = ? OR target_entity_id = ?',
                          (entity_id, entity_id))
            cursor.execute('DELETE FROM mentions WHERE entity_id = ?', (entity_id,))
            cursor.execute('DELETE FROM aliases WHERE entity_id = ?', (entity_id,))
            cursor.execute('DELETE FROM entities WHERE id = ?', (entity_id,))
            exp.conn.commit()

            return jsonify({
                'success': True,
                'action': 'delete',
                'message': f'Deleted entity "{row["canonical_name"]}"'
            })

        elif action == 'create_entity':
            name = parsed.get('name', '')
            entity_type = parsed.get('type', 'Organization')

            entity_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO entities (id, type, canonical_name, properties, confidence, status, created_at, updated_at)
                VALUES (?, ?, ?, '{}', 'confirmed', 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (entity_id, entity_type, name))
            exp.conn.commit()

            return jsonify({
                'success': True,
                'action': 'create_entity',
                'message': f'Created {entity_type} "{name}"',
                'entity_id': entity_id
            })

        elif action == 'create_edge':
            source_name = parsed.get('source', '')
            target_name = parsed.get('target', '')
            relation = parsed.get('relation', 'related_to')

            cursor.execute('SELECT id, canonical_name FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{source_name}%',))
            source_row = cursor.fetchone()

            cursor.execute('SELECT id, canonical_name FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{target_name}%',))
            target_row = cursor.fetchone()

            if not source_row:
                return jsonify({'success': False, 'error': f'Source "{source_name}" not found'})
            if not target_row:
                return jsonify({'success': False, 'error': f'Target "{target_name}" not found'})

            edge_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO edges (id, source_entity_id, target_entity_id, relation_type, confidence, properties, created_at)
                VALUES (?, ?, ?, ?, 'confirmed', '{}', CURRENT_TIMESTAMP)
            ''', (edge_id, source_row['id'], target_row['id'], relation))
            exp.conn.commit()

            return jsonify({
                'success': True,
                'action': 'create_edge',
                'message': f'Created: {source_row["canonical_name"]} --[{relation}]--> {target_row["canonical_name"]}'
            })

        elif action == 'change_type':
            entity_name = parsed.get('entity_name', '')
            new_type = parsed.get('new_type', '')

            cursor.execute('SELECT id, canonical_name, type FROM entities WHERE canonical_name LIKE ? LIMIT 1',
                          (f'%{entity_name}%',))
            row = cursor.fetchone()

            if not row:
                return jsonify({'success': False, 'error': f'Entity "{entity_name}" not found'})

            cursor.execute('UPDATE entities SET type = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                          (new_type, row['id']))
            exp.conn.commit()

            return jsonify({
                'success': True,
                'action': 'change_type',
                'message': f'Changed "{row["canonical_name"]}" from {row["type"]} to {new_type}'
            })

        else:
            return jsonify({'success': False, 'error': f'Unknown action: {action}'})

    except json.JSONDecodeError as e:
        return jsonify({'success': False, 'error': f'Failed to parse AI response: {str(e)}'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Duplicate Detection ====================

def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_duplicates(entities: List[Dict], threshold: float = 0.75) -> List[Dict]:
    """Find potential duplicate entities using fuzzy matching."""
    duplicates = []
    seen_pairs = set()

    # Group by type for faster comparison
    by_type: Dict[str, List[Dict]] = {}
    for e in entities:
        t = e.get('type', 'Unknown')
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    for entity_type, group in by_type.items():
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = group[i], group[j]
                name1 = e1.get('canonical_name', '')
                name2 = e2.get('canonical_name', '')

                # Skip if already seen
                pair_key = tuple(sorted([e1['id'], e2['id']]))
                if pair_key in seen_pairs:
                    continue

                # Calculate similarity
                sim = _similarity(name1, name2)
                if sim >= threshold:
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        'entity1': {
                            'id': e1['id'],
                            'name': name1,
                            'type': entity_type
                        },
                        'entity2': {
                            'id': e2['id'],
                            'name': name2,
                            'type': entity_type
                        },
                        'similarity': round(sim, 3),
                        'suggestion': name1 if len(name1) >= len(name2) else name2
                    })

    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    return duplicates


@api.route('/duplicates')
def api_find_duplicates():
    """Find potential duplicate entities for cleanup."""
    threshold = float(request.args.get('threshold', 0.75))
    limit = int(request.args.get('limit', 100))
    entity_type = request.args.get('type')  # Optional filter by type

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get all entities
        if entity_type:
            cursor.execute('''
                SELECT id, canonical_name, type
                FROM entities
                WHERE type = ?
                ORDER BY canonical_name
            ''', (entity_type,))
        else:
            cursor.execute('''
                SELECT id, canonical_name, type
                FROM entities
                ORDER BY type, canonical_name
            ''')

        entities = [dict(row) for row in cursor.fetchall()]

        # Find duplicates
        duplicates = _find_duplicates(entities, threshold)

        return jsonify({
            'total_entities': len(entities),
            'potential_duplicates': len(duplicates),
            'duplicates': duplicates[:limit],
            'threshold': threshold
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api.route('/batch-merge', methods=['POST'])
def api_batch_merge():
    """Merge multiple pairs of entities at once."""
    data = request.get_json()
    merges = data.get('merges', [])  # List of {keep_id, merge_id}

    if not merges:
        return jsonify({'error': 'No merges provided'}), 400

    try:
        kg = get_kg()
        results = []

        for merge in merges:
            keep_id = merge.get('keep_id')
            merge_id = merge.get('merge_id')

            if not keep_id or not merge_id:
                results.append({'success': False, 'error': 'Missing keep_id or merge_id'})
                continue

            try:
                kg.merge_entities(keep_id, merge_id)
                results.append({
                    'success': True,
                    'kept': keep_id,
                    'merged': merge_id
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'keep_id': keep_id,
                    'merge_id': merge_id,
                    'error': str(e)
                })

        successful = sum(1 for r in results if r.get('success'))
        return jsonify({
            'success': True,
            'total': len(merges),
            'successful': successful,
            'failed': len(merges) - successful,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Timeline ====================

def _parse_date(date_str: str) -> Optional[str]:
    """Try to parse a date string into ISO format for sorting."""
    import re
    from datetime import datetime

    if not date_str:
        return None

    # Common date patterns
    patterns = [
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),
        (r'(\d{1,2})/(\d{1,2})/(\d{2})', '%m/%d/%y'),
        (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', '%B %d %Y'),
        (r'(\d{1,2})\s+(\w+)\s+(\d{4})', '%d %B %Y'),
    ]

    # Try direct parsing
    for regex, fmt in patterns:
        match = re.search(regex, date_str)
        if match:
            try:
                date_part = match.group(0).replace(',', '')
                dt = datetime.strptime(date_part, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

    # Look for just a year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return f"{year_match.group(0)}-01-01"

    return None


@api.route('/timeline')
def api_timeline():
    """Get timeline of dated events/entities."""
    limit = int(request.args.get('limit', 200))

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get Date entities
        cursor.execute('''
            SELECT e.id, e.canonical_name, e.type, e.properties
            FROM entities e
            WHERE e.type = 'Date'
            ORDER BY e.canonical_name
        ''')
        date_entities = [dict(row) for row in cursor.fetchall()]

        # Get facts with dates (deadlines, key_terms with dates)
        cursor.execute('''
            SELECT e.id, e.canonical_name, e.type, e.properties
            FROM entities e
            WHERE e.type = 'Fact'
            AND (
                json_extract(e.properties, '$.fact_type') IN ('deadline', 'key_term')
                OR e.canonical_name LIKE '%date%'
                OR e.canonical_name LIKE '%deadline%'
            )
            LIMIT ?
        ''', (limit,))
        fact_entities = [dict(row) for row in cursor.fetchall()]

        # Build timeline entries
        timeline = []

        for entity in date_entities:
            parsed = _parse_date(entity['canonical_name'])
            props = json.loads(entity.get('properties', '{}') or '{}')

            # Get related entities
            cursor.execute('''
                SELECT DISTINCT e2.canonical_name, e2.type, ed.relation_type
                FROM edges ed
                JOIN entities e2 ON (
                    (ed.source_entity_id = ? AND ed.target_entity_id = e2.id)
                    OR (ed.target_entity_id = ? AND ed.source_entity_id = e2.id)
                )
                WHERE e2.type != 'Date'
                LIMIT 10
            ''', (entity['id'], entity['id']))
            related = [{'name': r['canonical_name'], 'type': r['type'], 'relation': r['relation_type']}
                      for r in cursor.fetchall()]

            timeline.append({
                'id': entity['id'],
                'date_raw': entity['canonical_name'],
                'date_parsed': parsed,
                'type': 'date',
                'description': entity['canonical_name'],
                'related_entities': related,
                'properties': props
            })

        for entity in fact_entities:
            props = json.loads(entity.get('properties', '{}') or '{}')
            parsed = _parse_date(props.get('due_date', '') or props.get('date', '') or entity['canonical_name'])

            timeline.append({
                'id': entity['id'],
                'date_raw': props.get('due_date', '') or props.get('date', ''),
                'date_parsed': parsed,
                'type': props.get('fact_type', 'fact'),
                'description': entity['canonical_name'][:200],
                'related_entities': [],
                'properties': props
            })

        # Sort by parsed date (entries without dates go at the end)
        timeline.sort(key=lambda x: (x['date_parsed'] is None, x['date_parsed'] or ''))

        return jsonify({
            'total': len(timeline),
            'timeline': timeline[:limit]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Export ====================

@api.route('/export')
def api_export():
    """Export graph data in various formats."""
    format_type = request.args.get('format', 'json')
    include_facts = request.args.get('include_facts', 'true').lower() == 'true'

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get all entities
        cursor.execute('''
            SELECT id, canonical_name, type, properties, confidence
            FROM entities
            WHERE type != 'Fact' OR ?
            ORDER BY type, canonical_name
        ''', (include_facts,))
        entities = [dict(row) for row in cursor.fetchall()]

        # Get all edges
        cursor.execute('''
            SELECT e.id, e.source_entity_id, e.target_entity_id, e.relation_type, e.properties,
                   src.canonical_name as source_name, tgt.canonical_name as target_name
            FROM edges e
            JOIN entities src ON e.source_entity_id = src.id
            JOIN entities tgt ON e.target_entity_id = tgt.id
        ''')
        edges = [dict(row) for row in cursor.fetchall()]

        if format_type == 'json':
            # Full JSON export
            return jsonify({
                'matter': _matter_name,
                'entities': entities,
                'edges': edges,
                'stats': {
                    'entity_count': len(entities),
                    'edge_count': len(edges)
                }
            })

        elif format_type == 'csv':
            # CSV format - entities and edges as separate files in a zip
            import io
            import csv
            import zipfile
            from flask import Response

            # Create zip in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Entities CSV
                entities_csv = io.StringIO()
                writer = csv.DictWriter(entities_csv, fieldnames=['id', 'canonical_name', 'type', 'properties', 'confidence'])
                writer.writeheader()
                writer.writerows(entities)
                zf.writestr('entities.csv', entities_csv.getvalue())

                # Edges CSV
                edges_csv = io.StringIO()
                writer = csv.DictWriter(edges_csv, fieldnames=['id', 'source_entity_id', 'target_entity_id', 'source_name', 'target_name', 'relation_type', 'properties'])
                writer.writeheader()
                writer.writerows(edges)
                zf.writestr('edges.csv', edges_csv.getvalue())

            zip_buffer.seek(0)
            return Response(
                zip_buffer.getvalue(),
                mimetype='application/zip',
                headers={'Content-Disposition': f'attachment; filename={_matter_name}_export.zip'}
            )

        elif format_type == 'graphml':
            # GraphML format for tools like Gephi, yEd
            from flask import Response

            graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
            graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
            graphml.append('  <key id="name" for="node" attr.name="name" attr.type="string"/>')
            graphml.append('  <key id="type" for="node" attr.name="type" attr.type="string"/>')
            graphml.append('  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>')
            graphml.append('  <graph id="G" edgedefault="directed">')

            # Add nodes
            for e in entities:
                eid = e['id'].replace('-', '_')
                name = e['canonical_name'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
                graphml.append(f'    <node id="{eid}">')
                graphml.append(f'      <data key="name">{name}</data>')
                graphml.append(f'      <data key="type">{e["type"]}</data>')
                graphml.append(f'    </node>')

            # Add edges
            for edge in edges:
                src = edge['source_entity_id'].replace('-', '_')
                tgt = edge['target_entity_id'].replace('-', '_')
                rel = edge['relation_type']
                graphml.append(f'    <edge source="{src}" target="{tgt}">')
                graphml.append(f'      <data key="relation">{rel}</data>')
                graphml.append(f'    </edge>')

            graphml.append('  </graph>')
            graphml.append('</graphml>')

            return Response(
                '\n'.join(graphml),
                mimetype='application/xml',
                headers={'Content-Disposition': f'attachment; filename={_matter_name}.graphml'}
            )

        else:
            return jsonify({'error': f'Unknown format: {format_type}'}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Graph Analytics ====================

def _compute_pagerank(adj: Dict[str, set], damping: float = 0.85, iterations: int = 100) -> Dict[str, float]:
    """Compute PageRank for all nodes."""
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}

    # Initialize PageRank
    pr = {node: 1.0 / n for node in nodes}

    for _ in range(iterations):
        new_pr = {}
        for node in nodes:
            rank = (1 - damping) / n
            for other in nodes:
                if node in adj.get(other, set()):
                    out_degree = len(adj.get(other, set()))
                    if out_degree > 0:
                        rank += damping * pr[other] / out_degree
            new_pr[node] = rank
        pr = new_pr

    return pr


def _compute_betweenness(adj: Dict[str, set], entity_ids: set, sample_size: int = 100) -> Dict[str, float]:
    """Compute betweenness centrality using sampling for large graphs."""
    from collections import deque

    nodes = list(entity_ids)
    n = len(nodes)
    if n == 0:
        return {}

    betweenness = {node: 0.0 for node in nodes}

    # Sample nodes for large graphs
    sample_nodes = nodes[:sample_size] if n > sample_size else nodes

    for source in sample_nodes:
        # BFS from source
        dist = {source: 0}
        paths = {source: 1}
        pred = {node: [] for node in nodes}
        queue = deque([source])
        visited_order = []

        while queue:
            v = queue.popleft()
            visited_order.append(v)
            for w in adj.get(v, set()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist.get(w) == dist[v] + 1:
                    paths[w] = paths.get(w, 0) + paths[v]
                    pred[w].append(v)

        # Accumulate betweenness
        delta = {node: 0.0 for node in nodes}
        for w in reversed(visited_order):
            for v in pred[w]:
                if paths.get(w, 0) > 0:
                    delta[v] += (paths.get(v, 0) / paths[w]) * (1 + delta[w])
            if w != source:
                betweenness[w] += delta[w]

    # Normalize
    if n > 2:
        norm = 2.0 / ((n - 1) * (n - 2))
        betweenness = {k: v * norm for k, v in betweenness.items()}

    return betweenness


@api.route('/analytics')
def api_analytics():
    """Compute graph analytics: degree centrality, PageRank, betweenness."""
    limit = int(request.args.get('limit', 50))
    metric = request.args.get('metric', 'all')  # degree, pagerank, betweenness, all

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get entities
        cursor.execute('SELECT id, canonical_name, type FROM entities')
        entities = {row['id']: dict(row) for row in cursor.fetchall()}

        # Get edges
        cursor.execute('SELECT source_entity_id, target_entity_id FROM edges')
        edges = [dict(row) for row in cursor.fetchall()]

        # Build adjacency
        adj: Dict[str, set] = {}
        in_degree: Dict[str, int] = {eid: 0 for eid in entities}
        out_degree: Dict[str, int] = {eid: 0 for eid in entities}

        for edge in edges:
            src = edge['source_entity_id']
            tgt = edge['target_entity_id']
            if src not in adj:
                adj[src] = set()
            adj[src].add(tgt)
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

        results = {}

        # Degree centrality (normalized)
        if metric in ['degree', 'all']:
            n = len(entities)
            degree_centrality = {}
            for eid in entities:
                total_degree = in_degree.get(eid, 0) + out_degree.get(eid, 0)
                degree_centrality[eid] = total_degree / (2 * (n - 1)) if n > 1 else 0

            sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            results['degree_centrality'] = [
                {
                    'entity_id': eid,
                    'name': entities[eid]['canonical_name'],
                    'type': entities[eid]['type'],
                    'score': round(score, 4),
                    'in_degree': in_degree.get(eid, 0),
                    'out_degree': out_degree.get(eid, 0)
                }
                for eid, score in sorted_degree[:limit]
            ]

        # PageRank
        if metric in ['pagerank', 'all']:
            # Build undirected adjacency for PageRank
            undirected_adj: Dict[str, set] = {}
            for edge in edges:
                src, tgt = edge['source_entity_id'], edge['target_entity_id']
                if src not in undirected_adj:
                    undirected_adj[src] = set()
                if tgt not in undirected_adj:
                    undirected_adj[tgt] = set()
                undirected_adj[src].add(tgt)
                undirected_adj[tgt].add(src)

            pagerank = _compute_pagerank(undirected_adj)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            results['pagerank'] = [
                {
                    'entity_id': eid,
                    'name': entities[eid]['canonical_name'],
                    'type': entities[eid]['type'],
                    'score': round(score, 6)
                }
                for eid, score in sorted_pr[:limit] if eid in entities
            ]

        # Betweenness centrality
        if metric in ['betweenness', 'all']:
            # Build undirected adjacency
            undirected_adj: Dict[str, set] = {}
            for edge in edges:
                src, tgt = edge['source_entity_id'], edge['target_entity_id']
                if src not in undirected_adj:
                    undirected_adj[src] = set()
                if tgt not in undirected_adj:
                    undirected_adj[tgt] = set()
                undirected_adj[src].add(tgt)
                undirected_adj[tgt].add(src)

            betweenness = _compute_betweenness(undirected_adj, set(entities.keys()))
            sorted_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            results['betweenness'] = [
                {
                    'entity_id': eid,
                    'name': entities[eid]['canonical_name'],
                    'type': entities[eid]['type'],
                    'score': round(score, 6)
                }
                for eid, score in sorted_bc[:limit] if eid in entities
            ]

        return jsonify({
            'total_entities': len(entities),
            'total_edges': len(edges),
            'analytics': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Shortest Path ====================

@api.route('/shortest-path')
def api_shortest_path():
    """Find shortest path between two entities."""
    source_id = request.args.get('source')
    target_id = request.args.get('target')

    if not source_id or not target_id:
        return jsonify({'error': 'source and target parameters required'}), 400

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get entities
        cursor.execute('SELECT id, canonical_name, type FROM entities')
        entities = {row['id']: dict(row) for row in cursor.fetchall()}

        if source_id not in entities:
            return jsonify({'error': f'Source entity {source_id} not found'}), 404
        if target_id not in entities:
            return jsonify({'error': f'Target entity {target_id} not found'}), 404

        # Get edges with relation info
        cursor.execute('''
            SELECT id, source_entity_id, target_entity_id, relation_type
            FROM edges
        ''')
        edges = [dict(row) for row in cursor.fetchall()]

        # Build undirected adjacency with edge info
        adj: Dict[str, List[Tuple[str, str, str]]] = {}  # node -> [(neighbor, edge_id, relation)]
        for edge in edges:
            src, tgt = edge['source_entity_id'], edge['target_entity_id']
            if src not in adj:
                adj[src] = []
            if tgt not in adj:
                adj[tgt] = []
            adj[src].append((tgt, edge['id'], edge['relation_type']))
            adj[tgt].append((src, edge['id'], edge['relation_type']))

        # BFS to find shortest path
        from collections import deque

        visited = {source_id: None}
        edge_used = {source_id: None}
        queue = deque([source_id])

        while queue:
            current = queue.popleft()
            if current == target_id:
                break

            for neighbor, edge_id, relation in adj.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    edge_used[neighbor] = (edge_id, relation)
                    queue.append(neighbor)

        if target_id not in visited:
            return jsonify({
                'found': False,
                'message': 'No path exists between these entities'
            })

        # Reconstruct path
        path_nodes = []
        path_edges = []
        current = target_id

        while current is not None:
            path_nodes.append({
                'id': current,
                'name': entities[current]['canonical_name'],
                'type': entities[current]['type']
            })
            if edge_used[current] is not None:
                edge_id, relation = edge_used[current]
                path_edges.append({
                    'id': edge_id,
                    'relation': relation
                })
            current = visited[current]

        path_nodes.reverse()
        path_edges.reverse()

        return jsonify({
            'found': True,
            'path_length': len(path_nodes) - 1,
            'nodes': path_nodes,
            'edges': path_edges
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Relationship Analysis ====================

@api.route('/relationship-analysis')
def api_relationship_analysis():
    """Analyze relationship patterns in the graph."""
    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get all edges with entity info
        cursor.execute('''
            SELECT e.relation_type,
                   e.source_entity_id, e.target_entity_id,
                   src.type as source_type, tgt.type as target_type
            FROM edges e
            LEFT JOIN entities src ON e.source_entity_id = src.id
            LEFT JOIN entities tgt ON e.target_entity_id = tgt.id
        ''')
        edges = [dict(row) for row in cursor.fetchall()]

        # Analyze relationship types
        relation_counts = {}
        relation_patterns = {}  # (source_type, relation, target_type) -> count

        for edge in edges:
            rel = edge['relation_type']
            src_type = edge['source_type'] or 'Unknown'
            tgt_type = edge['target_type'] or 'Unknown'

            relation_counts[rel] = relation_counts.get(rel, 0) + 1

            pattern = f"{src_type} -[{rel}]-> {tgt_type}"
            relation_patterns[pattern] = relation_patterns.get(pattern, 0) + 1

        # Sort by count
        sorted_relations = sorted(relation_counts.items(), key=lambda x: -x[1])
        sorted_patterns = sorted(relation_patterns.items(), key=lambda x: -x[1])

        # Find bidirectional relationships
        bidirectional_pairs = []
        edge_pairs = {}
        for edge in edges:
            src, tgt, rel = edge['source_entity_id'], edge['target_entity_id'], edge['relation_type']
            key = (min(src, tgt), max(src, tgt))
            if key not in edge_pairs:
                edge_pairs[key] = []
            edge_pairs[key].append((src, tgt, rel))

        for key, pair_edges in edge_pairs.items():
            if len(pair_edges) > 1:
                bidirectional_pairs.append({
                    'entity_pair': key,
                    'relationships': [{'source': e[0], 'target': e[1], 'relation': e[2]} for e in pair_edges]
                })

        # Get entity type distribution
        cursor.execute('''
            SELECT type, COUNT(*) as count
            FROM entities
            GROUP BY type
            ORDER BY count DESC
        ''')
        entity_type_dist = [dict(row) for row in cursor.fetchall()]

        return jsonify({
            'total_relationships': len(edges),
            'unique_relation_types': len(relation_counts),
            'relation_type_counts': [{'type': r, 'count': c} for r, c in sorted_relations],
            'top_patterns': [{'pattern': p, 'count': c} for p, c in sorted_patterns[:30]],
            'bidirectional_count': len(bidirectional_pairs),
            'entity_type_distribution': entity_type_dist
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Similar Entities (Embedding-based) ====================

@api.route('/similar/<entity_id>')
def api_similar_entities(entity_id):
    """Find entities similar to a given entity using embeddings."""
    limit = int(request.args.get('limit', 10))
    threshold = float(request.args.get('threshold', 0.5))

    try:
        kg = get_kg()
        exp = get_exporter()

        # Get the entity
        cursor = exp.conn.cursor()
        cursor.execute('SELECT id, canonical_name, type, properties FROM entities WHERE id = ?',
                      (entity_id,))
        entity_row = cursor.fetchone()

        if not entity_row:
            return jsonify({'error': 'Entity not found'}), 404

        entity = dict(entity_row)

        # Get similar entities from vector store
        from ..core.embeddings.vector_store import EmbeddingGenerator

        embedding_gen = EmbeddingGenerator()

        # Generate embedding for search
        search_text = f"{entity['canonical_name']} {entity['type']}"
        query_embedding = embedding_gen.generate_query_embedding(search_text)

        # Search vector store
        similar = kg.vector_store.search(query_embedding, k=limit + 1)  # +1 to exclude self

        # Get entity details for results
        similar_entities = []
        for sim_id, score in similar:
            if sim_id == entity_id:
                continue
            if score < threshold:
                continue

            cursor.execute('SELECT id, canonical_name, type, properties, confidence FROM entities WHERE id = ?',
                          (sim_id,))
            sim_row = cursor.fetchone()
            if sim_row:
                sim_entity = dict(sim_row)
                similar_entities.append({
                    'entity_id': sim_id,
                    'name': sim_entity['canonical_name'],
                    'type': sim_entity['type'],
                    'confidence': sim_entity['confidence'],
                    'similarity_score': round(score, 4)
                })

        return jsonify({
            'source_entity': {
                'id': entity_id,
                'name': entity['canonical_name'],
                'type': entity['type']
            },
            'similar_entities': similar_entities[:limit]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api.route('/similar-by-name')
def api_similar_by_name():
    """Find entities similar to a text query using embeddings."""
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 10))
    entity_type = request.args.get('type')

    if not query:
        return jsonify({'error': 'query parameter required'}), 400

    try:
        kg = get_kg()
        exp = get_exporter()

        from ..core.embeddings.vector_store import EmbeddingGenerator
        embedding_gen = EmbeddingGenerator()

        # Generate embedding for query
        search_text = f"{query} {entity_type}" if entity_type else query
        query_embedding = embedding_gen.generate_query_embedding(search_text)

        # Search vector store
        similar = kg.vector_store.search(query_embedding, k=limit * 2)  # Get extra for filtering

        # Get entity details
        cursor = exp.conn.cursor()
        results = []
        for sim_id, score in similar:
            cursor.execute('SELECT id, canonical_name, type, properties, confidence FROM entities WHERE id = ?',
                          (sim_id,))
            row = cursor.fetchone()
            if row:
                entity = dict(row)
                # Filter by type if specified
                if entity_type and entity['type'] != entity_type:
                    continue
                results.append({
                    'entity_id': sim_id,
                    'name': entity['canonical_name'],
                    'type': entity['type'],
                    'confidence': entity['confidence'],
                    'similarity_score': round(score, 4)
                })
                if len(results) >= limit:
                    break

        return jsonify({
            'query': query,
            'results': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Entity Importance Scoring ====================

@api.route('/importance')
def api_importance():
    """Score entities by importance using multiple metrics."""
    limit = int(request.args.get('limit', 30))
    entity_type = request.args.get('type')

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get entities
        if entity_type:
            cursor.execute('SELECT id, canonical_name, type, confidence FROM entities WHERE type = ?',
                          (entity_type,))
        else:
            cursor.execute('SELECT id, canonical_name, type, confidence FROM entities')
        entities = {row['id']: dict(row) for row in cursor.fetchall()}

        # Get edges
        cursor.execute('SELECT source_entity_id, target_entity_id, relation_type FROM edges')
        edges = [dict(row) for row in cursor.fetchall()]

        # Calculate metrics
        in_degree = {eid: 0 for eid in entities}
        out_degree = {eid: 0 for eid in entities}
        relation_diversity = {eid: set() for eid in entities}

        for edge in edges:
            src, tgt = edge['source_entity_id'], edge['target_entity_id']
            rel = edge['relation_type']
            if src in out_degree:
                out_degree[src] += 1
                relation_diversity[src].add(rel)
            if tgt in in_degree:
                in_degree[tgt] += 1
                relation_diversity[tgt].add(rel)

        # Get mention counts
        cursor.execute('''
            SELECT entity_id, COUNT(*) as mention_count
            FROM mentions
            GROUP BY entity_id
        ''')
        mention_counts = {row['entity_id']: row['mention_count'] for row in cursor.fetchall()}

        # Get alias counts (more aliases = more prominent)
        cursor.execute('''
            SELECT entity_id, COUNT(*) as alias_count
            FROM aliases
            GROUP BY entity_id
        ''')
        alias_counts = {row['entity_id']: row['alias_count'] for row in cursor.fetchall()}

        # Build undirected adjacency for PageRank
        adj = {}
        for edge in edges:
            src, tgt = edge['source_entity_id'], edge['target_entity_id']
            if src not in adj:
                adj[src] = set()
            if tgt not in adj:
                adj[tgt] = set()
            adj[src].add(tgt)
            adj[tgt].add(src)

        # Simplified PageRank
        n = len(entities)
        pr = {eid: 1.0 / n for eid in entities}
        damping = 0.85

        for _ in range(50):
            new_pr = {}
            for node in entities:
                rank = (1 - damping) / n
                for other in entities:
                    if node in adj.get(other, set()):
                        out_deg = len(adj.get(other, set()))
                        if out_deg > 0:
                            rank += damping * pr[other] / out_deg
                new_pr[node] = rank
            pr = new_pr

        # Calculate composite importance score
        importance_scores = []
        for eid, entity in entities.items():
            total_degree = in_degree.get(eid, 0) + out_degree.get(eid, 0)
            mentions = mention_counts.get(eid, 0)
            aliases = alias_counts.get(eid, 0)
            rel_types = len(relation_diversity.get(eid, set()))
            pagerank = pr.get(eid, 0)

            # Confidence boost
            conf_boost = 1.2 if entity['confidence'] == 'confirmed' else 1.0

            # Composite score (weighted combination)
            score = (
                0.25 * min(total_degree / 20, 1.0) +  # Degree (normalized to ~20)
                0.25 * pagerank * n +                  # PageRank (scaled up)
                0.20 * min(mentions / 10, 1.0) +       # Mentions
                0.15 * min(rel_types / 5, 1.0) +       # Relationship diversity
                0.15 * min(aliases / 3, 1.0)           # Aliases
            ) * conf_boost

            importance_scores.append({
                'entity_id': eid,
                'name': entity['canonical_name'],
                'type': entity['type'],
                'confidence': entity['confidence'],
                'importance_score': round(score, 4),
                'metrics': {
                    'in_degree': in_degree.get(eid, 0),
                    'out_degree': out_degree.get(eid, 0),
                    'total_degree': total_degree,
                    'pagerank': round(pagerank, 6),
                    'mention_count': mentions,
                    'alias_count': aliases,
                    'relation_type_count': rel_types
                }
            })

        # Sort by importance score
        importance_scores.sort(key=lambda x: x['importance_score'], reverse=True)

        return jsonify({
            'total_entities': len(entities),
            'rankings': importance_scores[:limit]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Temporal Queries ====================

@api.route('/temporal')
def api_temporal():
    """Query entities and facts by time range."""
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    query = request.args.get('query', '')

    try:
        from ..core.query.nl_query import NLQueryEngine

        kg = get_kg()
        query_engine = NLQueryEngine(kg.db, kg.vector_store)

        if query:
            # Parse natural language temporal query
            result = query_engine.query_by_timeframe(query)
        else:
            # Use explicit year range
            result = query_engine.temporal_query(start_year=start_year, end_year=end_year)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Find Connections ====================

@api.route('/connections')
def api_connections():
    """Find all paths/connections between two entities."""
    entity1 = request.args.get('entity1', '')
    entity2 = request.args.get('entity2', '')

    if not entity1 or not entity2:
        return jsonify({'error': 'Both entity1 and entity2 parameters required'}), 400

    try:
        from ..core.query.nl_query import NLQueryEngine

        kg = get_kg()
        query_engine = NLQueryEngine(kg.db, kg.vector_store)

        result = query_engine.find_connections(entity1, entity2)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Entity Clusters ====================

def _build_adjacency(edges: List[Dict]) -> Dict[str, set]:
    """Build adjacency list from edges."""
    adj: Dict[str, set] = {}
    for edge in edges:
        src = edge['source_entity_id']
        tgt = edge['target_entity_id']
        if src not in adj:
            adj[src] = set()
        if tgt not in adj:
            adj[tgt] = set()
        adj[src].add(tgt)
        adj[tgt].add(src)
    return adj


def _find_connected_components(adj: Dict[str, set], entity_ids: set) -> List[set]:
    """Find connected components using BFS."""
    visited = set()
    components = []

    for start in entity_ids:
        if start in visited:
            continue

        # BFS
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        if component:
            components.append(component)

    return components


@api.route('/clusters')
def api_clusters():
    """Find entity clusters (connected components) in the graph."""
    min_size = int(request.args.get('min_size', 3))
    entity_type = request.args.get('type')  # Optional filter

    try:
        exp = get_exporter()
        cursor = exp.conn.cursor()

        # Get entities
        if entity_type:
            cursor.execute('SELECT id, canonical_name, type FROM entities WHERE type = ?', (entity_type,))
        else:
            cursor.execute('SELECT id, canonical_name, type FROM entities')
        entities = {row['id']: dict(row) for row in cursor.fetchall()}

        # Get edges
        cursor.execute('SELECT source_entity_id, target_entity_id FROM edges')
        edges = [dict(row) for row in cursor.fetchall()]

        # Build adjacency and find components
        adj = _build_adjacency(edges)
        components = _find_connected_components(adj, set(entities.keys()))

        # Filter by min size and sort by size descending
        clusters = []
        for component in components:
            if len(component) >= min_size:
                cluster_entities = [entities[eid] for eid in component if eid in entities]
                # Group by type
                by_type: Dict[str, int] = {}
                for e in cluster_entities:
                    t = e['type']
                    by_type[t] = by_type.get(t, 0) + 1

                clusters.append({
                    'size': len(cluster_entities),
                    'entities': cluster_entities[:20],  # Limit to first 20 for display
                    'types': by_type,
                    'sample_names': [e['canonical_name'] for e in cluster_entities[:5]]
                })

        clusters.sort(key=lambda x: x['size'], reverse=True)

        return jsonify({
            'total_clusters': len(clusters),
            'total_entities': len(entities),
            'isolated_entities': sum(1 for c in components if len(c) < min_size),
            'clusters': clusters[:50]  # Top 50 clusters
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== App Factory ====================

def create_app(matter_name: str, api_key: str = GEMINI_API_KEY) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        matter_name: Name of the matter to load
        api_key: Gemini API key

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)

    # Initialize matter
    init_matter(matter_name, api_key)

    # Register API blueprint
    app.register_blueprint(api)

    return app
