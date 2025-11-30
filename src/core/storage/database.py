"""
SQLite database layer for the Knowledge Graph system.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .models import Entity, Edge, Mention, Document, Event, Alias


class Database:
    """SQLite database wrapper for knowledge graph storage."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._create_indexes()

    def _create_tables(self):
        """Create all required tables."""
        cursor = self.conn.cursor()

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence TEXT DEFAULT 'extracted',
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Aliases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aliases (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                alias_text TEXT NOT NULL,
                source TEXT DEFAULT 'extracted',
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
        """)

        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence TEXT DEFAULT 'extracted',
                provenance_doc_id TEXT,
                provenance_span TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
        """)

        # Mentions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mentions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                surface_text TEXT NOT NULL,
                context_snippet TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                added_at TEXT NOT NULL,
                processed_at TEXT
            )
        """)

        # Events table (event sourcing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload TEXT NOT NULL,
                user_initiated INTEGER DEFAULT 0
            )
        """)

        # Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                entity_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
        """)

        # Resolution queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resolution_queue (
                id TEXT PRIMARY KEY,
                mention_surface_text TEXT NOT NULL,
                mention_context TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                span_start INTEGER,
                span_end INTEGER,
                candidate_entities TEXT,  -- JSON array of {entity_id, score}
                status TEXT DEFAULT 'pending',  -- pending, resolved, skipped
                created_at TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def _create_indexes(self):
        """Create indexes for efficient queries."""
        cursor = self.conn.cursor()

        # Entity indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical_name ON entities(canonical_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type_confidence ON entities(type, confidence)")

        # Alias indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_entity_id ON aliases(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_alias_text ON aliases(alias_text)")

        # Edge indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_relation ON edges(source_entity_id, relation_type)")

        # Mention indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mentions_doc ON mentions(doc_id)")

        # Document indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(processed_at)")

        self.conn.commit()

    # ==================== Entity Operations ====================

    def add_entity(self, entity: Entity) -> str:
        """Add a new entity to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO entities (id, type, canonical_name, properties, confidence, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id, entity.type, entity.canonical_name,
            json.dumps(entity.properties), entity.confidence, entity.status,
            entity.created_at.isoformat(), entity.updated_at.isoformat()
        ))
        self.conn.commit()
        self._log_event("create_entity", {"entity_id": entity.id, "type": entity.type, "name": entity.canonical_name})
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_entity(row)
        return None

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Entity]:
        """Get entities by type."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE type = ? AND status = 'active' LIMIT ?",
                      (entity_type, limit))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def get_all_entities(self, limit: int = 1000) -> List[Entity]:
        """Get all active entities."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE status = 'active' LIMIT ?", (limit,))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def search_entities_by_name(self, name: str, limit: int = 20) -> List[Entity]:
        """Search entities by name (fuzzy match)."""
        cursor = self.conn.cursor()
        search_pattern = f"%{name}%"
        cursor.execute("""
            SELECT DISTINCT e.* FROM entities e
            LEFT JOIN aliases a ON e.id = a.entity_id
            WHERE (e.canonical_name LIKE ? OR a.alias_text LIKE ?)
            AND e.status = 'active'
            LIMIT ?
        """, (search_pattern, search_pattern, limit))
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def update_entity(self, entity: Entity):
        """Update an existing entity."""
        entity.updated_at = datetime.now()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE entities SET
                type = ?, canonical_name = ?, properties = ?, confidence = ?,
                status = ?, updated_at = ?
            WHERE id = ?
        """, (
            entity.type, entity.canonical_name, json.dumps(entity.properties),
            entity.confidence, entity.status, entity.updated_at.isoformat(), entity.id
        ))
        self.conn.commit()
        self._log_event("update_entity", {"entity_id": entity.id})

    def delete_entity(self, entity_id: str):
        """Soft delete an entity (mark as tombstone)."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE entities SET status = 'tombstone', updated_at = ? WHERE id = ?",
                      (datetime.now().isoformat(), entity_id))
        self.conn.commit()
        self._log_event("delete_entity", {"entity_id": entity_id})

    def merge_entities(self, keep_id: str, merge_id: str):
        """Merge two entities, keeping one and tombstoning the other."""
        cursor = self.conn.cursor()

        # Move all mentions from merge to keep
        cursor.execute("UPDATE mentions SET entity_id = ? WHERE entity_id = ?", (keep_id, merge_id))

        # Move all aliases from merge to keep
        cursor.execute("UPDATE aliases SET entity_id = ? WHERE entity_id = ?", (keep_id, merge_id))

        # Update edges - source
        cursor.execute("UPDATE edges SET source_entity_id = ? WHERE source_entity_id = ?", (keep_id, merge_id))

        # Update edges - target
        cursor.execute("UPDATE edges SET target_entity_id = ? WHERE target_entity_id = ?", (keep_id, merge_id))

        # Tombstone the merged entity
        cursor.execute("UPDATE entities SET status = 'tombstone', updated_at = ? WHERE id = ?",
                      (datetime.now().isoformat(), merge_id))

        self.conn.commit()
        self._log_event("merge_entities", {"keep_id": keep_id, "merge_id": merge_id}, user_initiated=True)

    def _row_to_entity(self, row) -> Entity:
        """Convert a database row to an Entity object."""
        return Entity(
            id=row["id"],
            type=row["type"],
            canonical_name=row["canonical_name"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            confidence=row["confidence"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    # ==================== Alias Operations ====================

    def add_alias(self, alias: Alias):
        """Add an alias for an entity."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO aliases (id, entity_id, alias_text, source)
            VALUES (?, ?, ?, ?)
        """, (alias.id, alias.entity_id, alias.alias_text, alias.source))
        self.conn.commit()

    def get_aliases(self, entity_id: str) -> List[Alias]:
        """Get all aliases for an entity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM aliases WHERE entity_id = ?", (entity_id,))
        return [Alias(
            id=row["id"],
            entity_id=row["entity_id"],
            alias_text=row["alias_text"],
            source=row["source"]
        ) for row in cursor.fetchall()]

    # ==================== Edge Operations ====================

    def add_edge(self, edge: Edge) -> str:
        """Add a new edge to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO edges (id, source_entity_id, target_entity_id, relation_type,
                             properties, confidence, provenance_doc_id, provenance_span, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge.id, edge.source_entity_id, edge.target_entity_id, edge.relation_type,
            json.dumps(edge.properties), edge.confidence, edge.provenance_doc_id,
            edge.provenance_span, edge.created_at.isoformat()
        ))
        self.conn.commit()
        self._log_event("create_edge", {
            "edge_id": edge.id,
            "source": edge.source_entity_id,
            "target": edge.target_entity_id,
            "relation": edge.relation_type
        })
        return edge.id

    def get_edges_from(self, entity_id: str) -> List[Edge]:
        """Get all outgoing edges from an entity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM edges WHERE source_entity_id = ?", (entity_id,))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_edges_to(self, entity_id: str) -> List[Edge]:
        """Get all incoming edges to an entity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM edges WHERE target_entity_id = ?", (entity_id,))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_all_edges(self, limit: int = 1000) -> List[Edge]:
        """Get all edges."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM edges LIMIT ?", (limit,))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> Dict[str, Any]:
        """Get neighboring entities within N hops."""
        visited = set()
        result = {"entities": [], "edges": []}

        def traverse(current_id: str, depth: int):
            if depth > max_hops or current_id in visited:
                return
            visited.add(current_id)

            entity = self.get_entity(current_id)
            if entity and entity.status == "active":
                result["entities"].append(entity)

            # Get outgoing edges
            for edge in self.get_edges_from(current_id):
                result["edges"].append(edge)
                traverse(edge.target_entity_id, depth + 1)

            # Get incoming edges
            for edge in self.get_edges_to(current_id):
                result["edges"].append(edge)
                traverse(edge.source_entity_id, depth + 1)

        traverse(entity_id, 0)
        return result

    def delete_edge(self, edge_id: str):
        """Delete an edge."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self.conn.commit()
        self._log_event("delete_edge", {"edge_id": edge_id})

    def _row_to_edge(self, row) -> Edge:
        """Convert a database row to an Edge object."""
        return Edge(
            id=row["id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            relation_type=row["relation_type"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            confidence=row["confidence"],
            provenance_doc_id=row["provenance_doc_id"],
            provenance_span=row["provenance_span"],
            created_at=datetime.fromisoformat(row["created_at"])
        )

    # ==================== Mention Operations ====================

    def add_mention(self, mention: Mention) -> str:
        """Add a new mention."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO mentions (id, entity_id, doc_id, span_start, span_end, surface_text, context_snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mention.id, mention.entity_id, mention.doc_id,
            mention.span_start, mention.span_end, mention.surface_text, mention.context_snippet
        ))
        self.conn.commit()
        return mention.id

    def get_mentions_for_entity(self, entity_id: str) -> List[Mention]:
        """Get all mentions of an entity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM mentions WHERE entity_id = ?", (entity_id,))
        return [self._row_to_mention(row) for row in cursor.fetchall()]

    def get_mentions_in_doc(self, doc_id: str) -> List[Mention]:
        """Get all mentions in a document."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM mentions WHERE doc_id = ?", (doc_id,))
        return [self._row_to_mention(row) for row in cursor.fetchall()]

    def _row_to_mention(self, row) -> Mention:
        """Convert a database row to a Mention object."""
        return Mention(
            id=row["id"],
            entity_id=row["entity_id"],
            doc_id=row["doc_id"],
            span_start=row["span_start"],
            span_end=row["span_end"],
            surface_text=row["surface_text"],
            context_snippet=row["context_snippet"]
        )

    # ==================== Document Operations ====================

    def add_document(self, document: Document) -> str:
        """Add a new document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO documents (id, filename, filepath, file_hash, added_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            document.id, document.filename, document.filepath, document.file_hash,
            document.added_at.isoformat(),
            document.processed_at.isoformat() if document.processed_at else None
        ))
        self.conn.commit()
        return document.id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get a document by file hash."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents")
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def mark_document_processed(self, doc_id: str):
        """Mark a document as processed."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE documents SET processed_at = ? WHERE id = ?",
                      (datetime.now().isoformat(), doc_id))
        self.conn.commit()

    def delete_document(self, doc_id: str):
        """Delete a document and all related data (cascade)."""
        cursor = self.conn.cursor()

        # Get entities that only exist because of this document
        cursor.execute("""
            SELECT entity_id FROM mentions WHERE doc_id = ?
            GROUP BY entity_id
        """, (doc_id,))
        entity_ids = [row["entity_id"] for row in cursor.fetchall()]

        # Delete mentions for this document
        cursor.execute("DELETE FROM mentions WHERE doc_id = ?", (doc_id,))

        # Delete edges with provenance from this document
        cursor.execute("DELETE FROM edges WHERE provenance_doc_id = ?", (doc_id,))

        # Check if any entities became orphaned (no remaining mentions)
        for entity_id in entity_ids:
            cursor.execute("SELECT COUNT(*) as cnt FROM mentions WHERE entity_id = ?", (entity_id,))
            if cursor.fetchone()["cnt"] == 0:
                # Orphaned - tombstone it
                cursor.execute("UPDATE entities SET status = 'tombstone', updated_at = ? WHERE id = ?",
                              (datetime.now().isoformat(), entity_id))

        # Delete the document
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        self._log_event("delete_document", {"doc_id": doc_id})

    def _row_to_document(self, row) -> Document:
        """Convert a database row to a Document object."""
        return Document(
            id=row["id"],
            filename=row["filename"],
            filepath=row["filepath"],
            file_hash=row["file_hash"],
            added_at=datetime.fromisoformat(row["added_at"]),
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None
        )

    # ==================== Resolution Queue Operations ====================

    def add_to_resolution_queue(self, surface_text: str, context: str, doc_id: str,
                                span_start: int, span_end: int, candidates: List[Dict]):
        """Add an unresolved mention to the queue."""
        cursor = self.conn.cursor()
        from .models import generate_id
        cursor.execute("""
            INSERT INTO resolution_queue (id, mention_surface_text, mention_context, doc_id,
                                         span_start, span_end, candidate_entities, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            generate_id(), surface_text, context, doc_id, span_start, span_end,
            json.dumps(candidates), datetime.now().isoformat()
        ))
        self.conn.commit()

    def get_pending_resolutions(self, limit: int = 50) -> List[Dict]:
        """Get pending resolution items."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM resolution_queue WHERE status = 'pending'
            ORDER BY created_at LIMIT ?
        """, (limit,))
        return [{
            "id": row["id"],
            "surface_text": row["mention_surface_text"],
            "context": row["mention_context"],
            "doc_id": row["doc_id"],
            "span_start": row["span_start"],
            "span_end": row["span_end"],
            "candidates": json.loads(row["candidate_entities"]) if row["candidate_entities"] else [],
            "created_at": row["created_at"]
        } for row in cursor.fetchall()]

    def resolve_queue_item(self, queue_id: str, entity_id: str):
        """Resolve a queue item by linking to an entity."""
        cursor = self.conn.cursor()

        # Get queue item details
        cursor.execute("SELECT * FROM resolution_queue WHERE id = ?", (queue_id,))
        row = cursor.fetchone()
        if not row:
            return

        # Create mention
        mention = Mention.create(
            entity_id=entity_id,
            doc_id=row["doc_id"],
            span_start=row["span_start"],
            span_end=row["span_end"],
            surface_text=row["mention_surface_text"],
            context_snippet=row["mention_context"]
        )
        self.add_mention(mention)

        # Mark as resolved
        cursor.execute("UPDATE resolution_queue SET status = 'resolved' WHERE id = ?", (queue_id,))
        self.conn.commit()

    # ==================== Event Logging ====================

    def _log_event(self, operation: str, payload: Dict, user_initiated: bool = False):
        """Log an event for audit trail."""
        event = Event.create(operation, payload, user_initiated)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO events (id, timestamp, operation, payload, user_initiated)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.id, event.timestamp.isoformat(), event.operation,
            json.dumps(event.payload), 1 if event.user_initiated else 0
        ))
        self.conn.commit()

    def get_events(self, limit: int = 100) -> List[Event]:
        """Get recent events."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [Event(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            operation=row["operation"],
            payload=json.loads(row["payload"]),
            user_initiated=bool(row["user_initiated"])
        ) for row in cursor.fetchall()]

    # ==================== Embedding Operations ====================

    def store_embedding(self, entity_id: str, vector: bytes):
        """Store an embedding vector for an entity."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (entity_id, vector)
            VALUES (?, ?)
        """, (entity_id, vector))
        self.conn.commit()

    def get_embedding(self, entity_id: str) -> Optional[bytes]:
        """Get embedding vector for an entity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT vector FROM embeddings WHERE entity_id = ?", (entity_id,))
        row = cursor.fetchone()
        return row["vector"] if row else None

    def get_all_embeddings(self) -> List[Tuple[str, bytes]]:
        """Get all embeddings."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT entity_id, vector FROM embeddings")
        return [(row["entity_id"], row["vector"]) for row in cursor.fetchall()]

    # ==================== Stats ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as cnt FROM entities WHERE status = 'active'")
        entity_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM edges")
        edge_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM documents")
        doc_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM mentions")
        mention_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM resolution_queue WHERE status = 'pending'")
        pending_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT type, COUNT(*) as cnt FROM entities WHERE status = 'active' GROUP BY type")
        type_counts = {row["type"]: row["cnt"] for row in cursor.fetchall()}

        return {
            "entities": entity_count,
            "edges": edge_count,
            "documents": doc_count,
            "mentions": mention_count,
            "pending_resolutions": pending_count,
            "entities_by_type": type_counts
        }

    def close(self):
        """Close database connection."""
        self.conn.close()
