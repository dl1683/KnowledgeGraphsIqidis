"""
Data models for the Knowledge Graph system.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import uuid


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


@dataclass
class Entity:
    """Represents a node in the knowledge graph."""
    id: str
    type: str  # Person, Organization, Document, Clause, Date, Money, Location, Reference, Fact
    canonical_name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "extracted"  # confirmed, extracted, inferred
    status: str = "active"  # active, pending_resolution, tombstone
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, type: str, canonical_name: str, properties: Dict[str, Any] = None,
               confidence: str = "extracted") -> "Entity":
        """Factory method to create a new entity."""
        return cls(
            id=generate_id(),
            type=type,
            canonical_name=canonical_name,
            properties=properties or {},
            confidence=confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "canonical_name": self.canonical_name,
            "properties": self.properties,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Alias:
    """Represents an alias/alternate name for an entity."""
    id: str
    entity_id: str
    alias_text: str
    source: str  # "extracted", "user", "defined_term"

    @classmethod
    def create(cls, entity_id: str, alias_text: str, source: str = "extracted") -> "Alias":
        return cls(
            id=generate_id(),
            entity_id=entity_id,
            alias_text=alias_text,
            source=source
        )


@dataclass
class Edge:
    """Represents a directed relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "extracted"
    provenance_doc_id: Optional[str] = None
    provenance_span: Optional[str] = None  # JSON: {"start": int, "end": int}
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, source_entity_id: str, target_entity_id: str, relation_type: str,
               properties: Dict[str, Any] = None, confidence: str = "extracted",
               provenance_doc_id: str = None, provenance_span: str = None) -> "Edge":
        return cls(
            id=generate_id(),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            provenance_doc_id=provenance_doc_id,
            provenance_span=provenance_span
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "provenance_doc_id": self.provenance_doc_id,
            "provenance_span": self.provenance_span,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Mention:
    """Represents a specific occurrence of an entity in a document."""
    id: str
    entity_id: str
    doc_id: str
    span_start: int
    span_end: int
    surface_text: str
    context_snippet: str  # Surrounding text for context

    @classmethod
    def create(cls, entity_id: str, doc_id: str, span_start: int, span_end: int,
               surface_text: str, context_snippet: str) -> "Mention":
        return cls(
            id=generate_id(),
            entity_id=entity_id,
            doc_id=doc_id,
            span_start=span_start,
            span_end=span_end,
            surface_text=surface_text,
            context_snippet=context_snippet
        )


@dataclass
class Document:
    """Represents a source document."""
    id: str
    filename: str
    filepath: str
    file_hash: str
    added_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None

    @classmethod
    def create(cls, filename: str, filepath: str, file_hash: str) -> "Document":
        return cls(
            id=generate_id(),
            filename=filename,
            filepath=filepath,
            file_hash=file_hash
        )


@dataclass
class Event:
    """Represents an operation on the graph (for event sourcing)."""
    id: str
    timestamp: datetime
    operation: str  # create_entity, merge_entities, split_entity, create_edge, delete_edge, etc.
    payload: Dict[str, Any]
    user_initiated: bool = False

    @classmethod
    def create(cls, operation: str, payload: Dict[str, Any], user_initiated: bool = False) -> "Event":
        return cls(
            id=generate_id(),
            timestamp=datetime.now(),
            operation=operation,
            payload=payload,
            user_initiated=user_initiated
        )
