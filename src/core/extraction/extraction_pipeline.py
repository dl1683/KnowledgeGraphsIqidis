"""
Complete extraction pipeline that orchestrates document processing.

Flow:
1. Parse document
2. Structural extraction (defined terms, parties)
3. Chunk document
4. Semantic extraction per chunk
5. Entity resolution
6. Store in graph
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..storage.database import Database
from ..storage.models import Entity, Edge, Mention, Document, Alias
from ..parsing.document_parser import DocumentParser, ParsedDocument
from ..parsing.chunker import Chunker, Chunk
from ..embeddings.vector_store import VectorStore, EmbeddingGenerator
from .structural_extractor import StructuralExtractor, StructuralExtraction
from .semantic_extractor import SemanticExtractor, SemanticExtraction, ExtractedEntity
from ..config import GEMINI_API_KEY, RESOLUTION_CONFIDENCE_THRESHOLD


class ExtractionPipeline:
    """Orchestrates the full extraction pipeline."""

    def __init__(self, db: Database, vector_store: VectorStore, api_key: str = GEMINI_API_KEY):
        self.db = db
        self.vector_store = vector_store
        self.parser = DocumentParser()
        self.chunker = Chunker()
        self.structural_extractor = StructuralExtractor()
        self.semantic_extractor = SemanticExtractor(api_key)
        self.embedding_generator = EmbeddingGenerator(api_key)

    def process_document(self, filepath: str, skip_if_exists: bool = True) -> Optional[str]:
        """Process a single document through the extraction pipeline.

        Returns the document ID if successful.
        """
        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print('='*60)

        # Step 1: Parse document
        print("\n[1/6] Parsing document...")
        parsed = self.parser.parse(filepath)
        if not parsed:
            print(f"Failed to parse: {filepath}")
            return None

        # Check if already processed
        if skip_if_exists:
            existing = self.db.get_document_by_hash(parsed.file_hash)
            if existing and existing.processed_at:
                print(f"Document already processed: {existing.id}")
                return existing.id

        # Create document record
        doc = Document.create(
            filename=parsed.filename,
            filepath=parsed.filepath,
            file_hash=parsed.file_hash
        )
        doc_id = self.db.add_document(doc)
        print(f"Created document record: {doc_id}")

        # Step 2: Structural extraction
        print("\n[2/6] Extracting structural elements...")
        structural = self.structural_extractor.extract(parsed.text)
        self._store_structural_results(structural, doc_id, parsed.text)

        # Step 3: Chunk document
        print("\n[3/6] Chunking document...")
        chunks = self.chunker.chunk_text(parsed.text)
        print(f"Created {len(chunks)} chunks")

        # Get existing entities for context
        existing_entities = [e.canonical_name for e in self.db.get_all_entities(limit=100)]

        # Step 4: Semantic extraction per chunk
        print("\n[4/6] Extracting entities, relations, and facts...")
        all_entities = []
        all_relations = []
        all_facts = []

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...", end=" ")

            try:
                extraction = self.semantic_extractor.extract(chunk.text, existing_entities)

                # Add span offsets to entities
                for entity in extraction.entities:
                    entity.properties['chunk_start'] = chunk.start_char
                    entity.properties['chunk_end'] = chunk.end_char

                all_entities.extend(extraction.entities)
                all_relations.extend(extraction.relations)
                all_facts.extend(extraction.facts)

                # Update existing entities for next chunk
                existing_entities.extend([e.name for e in extraction.entities])

                print(f"Found {len(extraction.entities)} entities, {len(extraction.relations)} relations, {len(extraction.facts)} facts")

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Step 5: Entity resolution and storage
        print("\n[5/6] Resolving and storing entities...")
        entity_map = self._resolve_and_store_entities(all_entities, doc_id, parsed.text)

        # Step 6: Store relations and facts
        print("\n[6/6] Storing relations and facts...")
        self._store_relations(all_relations, entity_map, doc_id)
        self._store_facts(all_facts, entity_map, doc_id)

        # Mark document as processed
        self.db.mark_document_processed(doc_id)

        # Save vector store
        self.vector_store.save()

        print(f"\n{'='*60}")
        print(f"Document processing complete: {doc_id}")
        stats = self.db.get_stats()
        print(f"Graph now has: {stats['entities']} entities, {stats['edges']} edges")
        print('='*60)

        return doc_id

    def process_directory(self, dirpath: str, recursive: bool = True) -> List[str]:
        """Process all documents in a directory.

        Returns list of successfully processed document IDs.
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            print(f"Directory not found: {dirpath}")
            return []

        supported_extensions = self.parser.get_supported_extensions()
        processed_ids = []

        # Find all files
        if recursive:
            files = []
            for ext in supported_extensions:
                files.extend(dirpath.rglob(f"*{ext}"))
        else:
            files = []
            for ext in supported_extensions:
                files.extend(dirpath.glob(f"*{ext}"))

        print(f"Found {len(files)} documents to process")

        for filepath in files:
            try:
                doc_id = self.process_document(str(filepath))
                if doc_id:
                    processed_ids.append(doc_id)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        return processed_ids

    def _store_structural_results(self, structural: StructuralExtraction, doc_id: str, full_text: str):
        """Store results from structural extraction."""
        # Store parties as entities
        for party in structural.parties:
            entity = Entity.create(
                type="Organization" if any(corp in party.name for corp in ['Inc', 'Corp', 'LLC', 'Ltd', 'LLP']) else "Person",
                canonical_name=party.name,
                properties={
                    "role": party.role,
                    "source": "structural"
                },
                confidence="confirmed"  # Structural extractions are high confidence
            )
            entity_id = self.db.add_entity(entity)

            # Add aliases
            for alias in party.aliases:
                if alias != party.name:
                    self.db.add_alias(Alias.create(entity_id, alias, "defined_term"))

            # Add mention
            mention = Mention.create(
                entity_id=entity_id,
                doc_id=doc_id,
                span_start=party.span_start,
                span_end=party.span_end,
                surface_text=party.name,
                context_snippet=full_text[max(0, party.span_start-100):party.span_end+100]
            )
            self.db.add_mention(mention)

            # Generate and store embedding
            self._store_entity_embedding(entity_id, party.name, party.role)

        # Store defined terms
        for term in structural.defined_terms:
            entity = Entity.create(
                type="Reference",
                canonical_name=term.term,
                properties={
                    "definition": term.definition,
                    "source": "structural"
                },
                confidence="confirmed"
            )
            entity_id = self.db.add_entity(entity)

            # Add aliases
            for alias in term.aliases:
                if alias != term.term:
                    self.db.add_alias(Alias.create(entity_id, alias, "defined_term"))

            # Generate embedding
            self._store_entity_embedding(entity_id, term.term, term.definition)

        # Store document metadata as entity
        if structural.document_type != 'unknown':
            doc_entity = Entity.create(
                type="Document",
                canonical_name=f"Doc_{doc_id[:8]}",
                properties={
                    "document_type": structural.document_type,
                    "case_number": structural.case_number,
                    "court": structural.court_or_tribunal,
                    "source": "structural"
                },
                confidence="confirmed"
            )
            self.db.add_entity(doc_entity)

        print(f"  Stored {len(structural.parties)} parties, {len(structural.defined_terms)} defined terms")

    def _resolve_and_store_entities(self, entities: List[ExtractedEntity], doc_id: str, full_text: str) -> Dict[str, str]:
        """Resolve extracted entities against existing graph and store.

        Returns mapping of entity name -> entity ID.
        """
        entity_map = {}

        for entity in entities:
            if not entity.name or len(entity.name) < 2:
                continue

            # Check if entity already exists (by name match)
            existing = self.db.search_entities_by_name(entity.name, limit=5)

            if existing:
                # Check for exact match
                exact_match = None
                for e in existing:
                    if e.canonical_name.lower() == entity.name.lower():
                        exact_match = e
                        break

                if exact_match:
                    entity_map[entity.name] = exact_match.id
                    continue

                # Try embedding similarity for fuzzy match
                if self.vector_store.get_count() > 0:
                    query_embedding = self.embedding_generator.generate_query_embedding(
                        f"{entity.name} {entity.type}"
                    )
                    similar = self.vector_store.search(query_embedding, k=3)

                    for sim_id, score in similar:
                        if score > RESOLUTION_CONFIDENCE_THRESHOLD:
                            sim_entity = self.db.get_entity(sim_id)
                            if sim_entity and sim_entity.type == entity.type:
                                # High confidence match - link to existing
                                entity_map[entity.name] = sim_id
                                # Add as alias
                                self.db.add_alias(Alias.create(sim_id, entity.name, "extracted"))
                                break
                    else:
                        # No high confidence match - add to resolution queue or create new
                        if similar and similar[0][1] > 0.5:
                            # Medium confidence - queue for review
                            self.db.add_to_resolution_queue(
                                surface_text=entity.name,
                                context=entity.span_text[:200],
                                doc_id=doc_id,
                                span_start=entity.properties.get('chunk_start', 0),
                                span_end=entity.properties.get('chunk_end', 0),
                                candidates=[{"entity_id": eid, "score": float(s)} for eid, s in similar[:3]]
                            )
                            # Still create new entity (user can merge later)
                            new_id = self._create_new_entity(entity, doc_id, full_text)
                            entity_map[entity.name] = new_id
                        else:
                            # Low confidence - create new
                            new_id = self._create_new_entity(entity, doc_id, full_text)
                            entity_map[entity.name] = new_id
                else:
                    # No embeddings yet - use simple matching
                    new_id = self._create_new_entity(entity, doc_id, full_text)
                    entity_map[entity.name] = new_id
            else:
                # No existing entity - create new
                new_id = self._create_new_entity(entity, doc_id, full_text)
                entity_map[entity.name] = new_id

        return entity_map

    def _create_new_entity(self, entity: ExtractedEntity, doc_id: str, full_text: str) -> str:
        """Create a new entity in the database."""
        db_entity = Entity.create(
            type=entity.type,
            canonical_name=entity.name,
            properties=entity.properties,
            confidence="extracted"
        )
        entity_id = self.db.add_entity(db_entity)

        # Add mention
        chunk_start = entity.properties.get('chunk_start', 0)
        chunk_end = entity.properties.get('chunk_end', len(full_text))

        mention = Mention.create(
            entity_id=entity_id,
            doc_id=doc_id,
            span_start=chunk_start,
            span_end=chunk_end,
            surface_text=entity.span_text[:200],
            context_snippet=full_text[chunk_start:min(chunk_start+300, len(full_text))]
        )
        self.db.add_mention(mention)

        # Generate and store embedding
        self._store_entity_embedding(entity_id, entity.name, str(entity.properties))

        return entity_id

    def _store_entity_embedding(self, entity_id: str, name: str, context: str = ""):
        """Generate and store embedding for an entity."""
        try:
            text = f"{name} {context}"[:500]
            embedding = self.embedding_generator.generate(text)
            self.vector_store.add(entity_id, embedding)
        except Exception as e:
            print(f"Warning: Could not generate embedding for {name}: {e}")

    def _store_relations(self, relations: List, entity_map: Dict[str, str], doc_id: str):
        """Store extracted relations in the graph."""
        stored = 0

        for rel in relations:
            source_id = entity_map.get(rel.source_name)
            target_id = entity_map.get(rel.target_name)

            if not source_id or not target_id:
                # Try fuzzy match
                source_id = self._find_entity_by_name(rel.source_name, entity_map)
                target_id = self._find_entity_by_name(rel.target_name, entity_map)

            if source_id and target_id:
                edge = Edge.create(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=rel.relation_type,
                    properties=rel.properties,
                    confidence="extracted",
                    provenance_doc_id=doc_id
                )
                self.db.add_edge(edge)
                stored += 1

        print(f"  Stored {stored} relations")

    def _store_facts(self, facts: List, entity_map: Dict[str, str], doc_id: str):
        """Store extracted facts as Fact entities."""
        stored = 0

        for fact in facts:
            # Create Fact entity
            fact_entity = Entity.create(
                type="Fact",
                canonical_name=f"{fact.fact_type}: {fact.text[:50]}...",
                properties={
                    "fact_type": fact.fact_type,
                    "full_text": fact.text,
                    **fact.properties
                },
                confidence="extracted"
            )
            fact_id = self.db.add_entity(fact_entity)

            # Link to related entities
            for entity_name in fact.related_entities:
                entity_id = entity_map.get(entity_name) or self._find_entity_by_name(entity_name, entity_map)
                if entity_id:
                    edge = Edge.create(
                        source_entity_id=fact_id,
                        target_entity_id=entity_id,
                        relation_type="about",
                        properties={},
                        confidence="extracted",
                        provenance_doc_id=doc_id
                    )
                    self.db.add_edge(edge)

            stored += 1

        print(f"  Stored {stored} facts")

    def _find_entity_by_name(self, name: str, entity_map: Dict[str, str]) -> Optional[str]:
        """Find entity ID by name with fuzzy matching."""
        # Exact match
        if name in entity_map:
            return entity_map[name]

        # Case-insensitive match
        name_lower = name.lower()
        for k, v in entity_map.items():
            if k.lower() == name_lower:
                return v

        # Partial match
        for k, v in entity_map.items():
            if name_lower in k.lower() or k.lower() in name_lower:
                return v

        # Database search
        existing = self.db.search_entities_by_name(name, limit=1)
        if existing:
            return existing[0].id

        return None
