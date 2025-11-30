"""
Main Knowledge Graph interface.

Provides a unified interface for:
- Document processing
- Natural language queries
- Graph editing
- Visualization data
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from .storage.database import Database
from .storage.models import Entity, Edge, Mention, Alias
from .embeddings.vector_store import VectorStore
from .extraction.extraction_pipeline import ExtractionPipeline
from .query.nl_query import NLQueryEngine, QueryResult
from .config import GEMINI_API_KEY, MATTERS_DIR


class KnowledgeGraph:
    """Main interface for the Knowledge Graph system."""

    def __init__(self, matter_name: str, api_key: str = GEMINI_API_KEY):
        """Initialize a knowledge graph for a specific matter.

        Args:
            matter_name: Name/ID of the matter (used for storage paths)
            api_key: Gemini API key
        """
        self.matter_name = matter_name
        self.api_key = api_key

        # Set up paths
        self.matter_dir = MATTERS_DIR / matter_name
        self.matter_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.db = Database(str(self.matter_dir / "graph.db"))
        self.vector_store = VectorStore(str(self.matter_dir))
        self.extraction_pipeline = ExtractionPipeline(self.db, self.vector_store, api_key)
        self.query_engine = NLQueryEngine(self.db, self.vector_store, api_key)

    # ==================== Document Processing ====================

    def add_document(self, filepath: str) -> Optional[str]:
        """Add and process a single document.

        Returns document ID if successful.
        """
        return self.extraction_pipeline.process_document(filepath)

    def add_documents_from_directory(self, dirpath: str, recursive: bool = True) -> List[str]:
        """Add and process all documents from a directory.

        Returns list of processed document IDs.
        """
        return self.extraction_pipeline.process_directory(dirpath, recursive)

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents."""
        docs = self.db.get_all_documents()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "filepath": d.filepath,
                "added_at": d.added_at.isoformat(),
                "processed_at": d.processed_at.isoformat() if d.processed_at else None
            }
            for d in docs
        ]

    def remove_document(self, doc_id: str):
        """Remove a document and all related data."""
        self.db.delete_document(doc_id)

    # ==================== Querying ====================

    def query(self, question: str) -> QueryResult:
        """Ask a natural language question about the knowledge graph.

        Returns a QueryResult with entities, relationships, and a generated answer.
        """
        return self.query_engine.query(question)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific entity."""
        entity = self.db.get_entity(entity_id)
        if not entity:
            return None

        return {
            "id": entity.id,
            "name": entity.canonical_name,
            "type": entity.type,
            "properties": entity.properties,
            "confidence": entity.confidence,
            "status": entity.status
        }

    def get_entity_summary(self, entity_name: str) -> str:
        """Get a comprehensive summary of an entity."""
        return self.query_engine.get_entity_summary(entity_name)

    def search_entities(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for entities by name."""
        entities = self.db.search_entities_by_name(name, limit)
        return [
            {
                "id": e.id,
                "name": e.canonical_name,
                "type": e.type,
                "confidence": e.confidence
            }
            for e in entities
        ]

    def list_entities(self, entity_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List all entities, optionally filtered by type."""
        return self.query_engine.list_entities(entity_type, limit)

    def get_entity_neighbors(self, entity_id: str, hops: int = 1) -> Dict[str, Any]:
        """Get neighboring entities within N hops."""
        result = self.db.get_entity_neighbors(entity_id, max_hops=hops)
        return {
            "entities": [e.to_dict() for e in result["entities"]],
            "edges": [e.to_dict() for e in result["edges"]]
        }

    # ==================== Graph Editing ====================

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Merge two entities, keeping one and removing the other.

        Returns True if successful.
        """
        try:
            self.db.merge_entities(keep_id, merge_id)
            return True
        except Exception as e:
            print(f"Error merging entities: {e}")
            return False

    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                 properties: Dict[str, Any] = None) -> Optional[str]:
        """Add a new edge between entities.

        Returns edge ID if successful.
        """
        try:
            edge = Edge.create(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=relation_type,
                properties=properties or {},
                confidence="confirmed"
            )
            return self.db.add_edge(edge)
        except Exception as e:
            print(f"Error adding edge: {e}")
            return None

    def remove_edge(self, edge_id: str):
        """Remove an edge from the graph."""
        self.db.delete_edge(edge_id)

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity properties.

        Args:
            entity_id: ID of entity to update
            updates: Dict with optional keys: canonical_name, type, properties, confidence
        """
        try:
            entity = self.db.get_entity(entity_id)
            if not entity:
                return False

            if 'canonical_name' in updates:
                entity.canonical_name = updates['canonical_name']
            if 'type' in updates:
                entity.type = updates['type']
            if 'properties' in updates:
                entity.properties.update(updates['properties'])
            if 'confidence' in updates:
                entity.confidence = updates['confidence']

            self.db.update_entity(entity)
            return True
        except Exception as e:
            print(f"Error updating entity: {e}")
            return False

    def add_entity_alias(self, entity_id: str, alias: str) -> bool:
        """Add an alias for an entity."""
        try:
            self.db.add_alias(Alias.create(entity_id, alias, "user"))
            return True
        except Exception as e:
            print(f"Error adding alias: {e}")
            return False

    # ==================== Resolution Queue ====================

    def get_pending_resolutions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending entity resolutions for user review."""
        return self.db.get_pending_resolutions(limit)

    def resolve_entity(self, queue_id: str, entity_id: str):
        """Resolve a pending entity to an existing entity."""
        self.db.resolve_queue_item(queue_id, entity_id)

    # ==================== Statistics and Export ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self.db.get_stats()

    def get_visualization_data(self, center_entity_id: str = None, max_nodes: int = 100) -> Dict[str, Any]:
        """Get data formatted for graph visualization.

        Returns nodes and edges in a format suitable for D3.js or similar.
        """
        if center_entity_id:
            # Get subgraph around center entity
            result = self.db.get_entity_neighbors(center_entity_id, max_hops=2)
            entities = result["entities"][:max_nodes]
            edges = result["edges"]
        else:
            # Get top entities by connections
            entities = self.db.get_all_entities(limit=max_nodes)
            edges = self.db.get_all_edges(limit=500)

        # Filter edges to only include visible entities
        entity_ids = {e.id for e in entities}
        edges = [e for e in edges if e.source_entity_id in entity_ids and e.target_entity_id in entity_ids]

        # Format for visualization
        nodes = [
            {
                "id": e.id,
                "label": e.canonical_name[:30],
                "type": e.type,
                "confidence": e.confidence,
                "properties": e.properties
            }
            for e in entities
        ]

        links = [
            {
                "source": e.source_entity_id,
                "target": e.target_entity_id,
                "type": e.relation_type,
                "confidence": e.confidence
            }
            for e in edges
        ]

        return {
            "nodes": nodes,
            "links": links,
            "stats": self.get_stats()
        }

    def export_graph(self, filepath: str):
        """Export the entire graph to a JSON file."""
        import json

        data = {
            "matter": self.matter_name,
            "stats": self.get_stats(),
            "entities": [e.to_dict() for e in self.db.get_all_entities(limit=10000)],
            "edges": [e.to_dict() for e in self.db.get_all_edges(limit=50000)],
            "documents": self.get_documents()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def close(self):
        """Close database connections and save state."""
        self.vector_store.save()
        self.db.close()
