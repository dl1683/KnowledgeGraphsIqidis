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
from typing import Optional
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
