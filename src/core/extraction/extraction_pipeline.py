"""
Complete extraction pipeline that orchestrates document processing.

Flow:
1. Parse document
2. Structural extraction (defined terms, parties)
3. Chunk document
4. Semantic extraction per chunk (PARALLEL)
5. Entity resolution
6. Store in graph

Performance optimizations:
- Large chunks (20K tokens) for fewer API calls
- Unified extraction (1 API call per chunk instead of 3)
- Parallel chunk processing with ThreadPoolExecutor
- Global rate limiter for API safety
"""
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def _print(*args, **kwargs):
    """Print with flush for real-time output."""
    print(*args, **kwargs, flush=True)

from ..storage.database import Database
from ..storage.models import Entity, Edge, Mention, Document, Alias
from ..parsing.document_parser import DocumentParser, ParsedDocument
from ..parsing.chunker import Chunker, Chunk
from ..embeddings.vector_store import VectorStore, EmbeddingGenerator
from .structural_extractor import StructuralExtractor, StructuralExtraction
from .semantic_extractor import SemanticExtractor, SemanticExtraction, ExtractedEntity, RelationshipInferrer
from ..config import GEMINI_API_KEY, RESOLUTION_CONFIDENCE_THRESHOLD

# Parallel processing settings
MAX_PARALLEL_WORKERS = 3  # Stay within rate limits (15 req/min ÷ 4s = ~3.75)
REQUESTS_PER_MINUTE = 15
MIN_DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE


class GlobalRateLimiter:
    """Thread-safe rate limiter for parallel API calls."""

    def __init__(self, requests_per_minute: int = REQUESTS_PER_MINUTE):
        self.min_delay = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_request_time = 0

    def acquire(self):
        """Acquire rate limit slot, blocking if necessary."""
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Global rate limiter instance
_global_rate_limiter = GlobalRateLimiter()


class EntityNormalizer:
    """Normalize and match entity names for better coreference resolution."""

    # Common suffixes to strip for matching
    ORG_SUFFIXES = [
        ', Inc.', ', Inc', ' Inc.', ' Inc', ' LLC', ' L.L.C.', ' LLP', ' L.L.P.',
        ', Ltd.', ', Ltd', ' Ltd.', ' Ltd', ' Corp.', ' Corp', ' Corporation',
        ' Co.', ' Co', ' Company', ' & Co.', ' & Co', ' PLC', ' plc',
        ' Limited', ' Incorporated', ' Associates', ' & Associates',
        ' Partners', ' & Partners', ' Group', ' Holdings', ' International',
    ]

    # Common title prefixes for people
    PERSON_PREFIXES = [
        'Mr. ', 'Mrs. ', 'Ms. ', 'Miss ', 'Dr. ', 'Prof. ', 'Professor ',
        'Hon. ', 'Honorable ', 'Judge ', 'Justice ', 'Sen. ', 'Senator ',
        'Rep. ', 'Representative ', 'Atty. ', 'Attorney ', 'Esq.',
    ]

    # Common abbreviation mappings
    ABBREVIATIONS = {
        'intl': 'international',
        'int\'l': 'international',
        'natl': 'national',
        'nat\'l': 'national',
        'corp': 'corporation',
        'assoc': 'associates',
        'mgmt': 'management',
        'svcs': 'services',
        'svc': 'service',
        'tech': 'technology',
        'sys': 'systems',
        'grp': 'group',
        'hldgs': 'holdings',
        'mfg': 'manufacturing',
        'dist': 'distribution',
        'dev': 'development',
    }

    @staticmethod
    def normalize_org_name(name: str) -> str:
        """Normalize organization name for matching."""
        normalized = name.strip()

        # Remove common suffixes
        for suffix in EntityNormalizer.ORG_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
            elif normalized.lower().endswith(suffix.lower()):
                normalized = normalized[:-len(suffix)].strip()

        # Expand abbreviations
        words = normalized.split()
        expanded = []
        for word in words:
            word_lower = word.lower().rstrip('.,')
            if word_lower in EntityNormalizer.ABBREVIATIONS:
                expanded.append(EntityNormalizer.ABBREVIATIONS[word_lower])
            else:
                expanded.append(word)
        normalized = ' '.join(expanded)

        return normalized.strip()

    @staticmethod
    def normalize_person_name(name: str) -> str:
        """Normalize person name for matching."""
        normalized = name.strip()

        # Remove title prefixes
        for prefix in EntityNormalizer.PERSON_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
            elif normalized.lower().startswith(prefix.lower()):
                normalized = normalized[len(prefix):].strip()

        # Remove trailing suffixes like Jr., Sr., III, Esq.
        suffixes = [', Jr.', ', Jr', ' Jr.', ' Jr', ', Sr.', ', Sr', ' Sr.', ' Sr',
                   ', III', ' III', ', II', ' II', ', IV', ' IV', ', Esq.', ', Esq', ' Esq.', ' Esq']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()

        return normalized.strip()

    # Entity type indicators for validation
    ORG_TYPE_INDICATORS = [
        'corporation', 'corp.', 'corp', 'incorporated', 'inc.', 'inc',
        'limited', 'ltd.', 'ltd', 'llc', 'l.l.c.', 'llp', 'l.l.p.',
        'company', 'co.', 'holdings', 'group', 'partners', 'associates',
        'enterprises', 'industries', 'international', 'solutions',
        'services', 'systems', 'technologies', 'aerospace', 'aviation',
        'foundation', 'institute', 'association', 'plc', 'gmbh', 'ag',
    ]

    @staticmethod
    def validate_entity_type(name: str, claimed_type: str) -> str:
        """
        Validate and potentially correct the entity type based on name patterns.

        Returns the corrected type if the claimed type appears incorrect.
        """
        name_lower = name.lower()

        # Check if the name clearly indicates an Organization
        for indicator in EntityNormalizer.ORG_TYPE_INDICATORS:
            if indicator in name_lower:
                # Name contains organization indicator but was classified as Person
                if claimed_type == 'Person':
                    return 'Organization'
                return claimed_type

        # Check for person name patterns (Mr., Dr., etc.)
        for prefix in EntityNormalizer.PERSON_PREFIXES:
            if name_lower.startswith(prefix.lower()):
                if claimed_type == 'Organization':
                    return 'Person'
                return claimed_type

        # Check for ALL CAPS names that might be organizations
        if name.isupper() and len(name) > 3:
            words = name.split()
            # Multiple uppercase words often indicate organization
            if len(words) > 1 and any(ind in name_lower for ind in ['aerospace', 'corp', 'inc', 'ltd']):
                if claimed_type == 'Person':
                    return 'Organization'

        return claimed_type

    @staticmethod
    def normalize_name(name: str, entity_type: str = None) -> str:
        """Normalize entity name based on type."""
        if entity_type == 'Organization':
            return EntityNormalizer.normalize_org_name(name)
        elif entity_type == 'Person':
            return EntityNormalizer.normalize_person_name(name)
        else:
            return name.strip()

    @staticmethod
    def compute_similarity(name1: str, name2: str, entity_type: str = None) -> float:
        """Compute similarity score between two entity names."""
        from difflib import SequenceMatcher

        # Normalize both names
        n1 = EntityNormalizer.normalize_name(name1, entity_type).lower()
        n2 = EntityNormalizer.normalize_name(name2, entity_type).lower()

        # Exact match after normalization
        if n1 == n2:
            return 1.0

        # Check if one contains the other (partial match)
        if n1 in n2 or n2 in n1:
            shorter = min(len(n1), len(n2))
            longer = max(len(n1), len(n2))
            return 0.7 + (0.3 * shorter / longer)

        # Check word overlap for organizations
        if entity_type == 'Organization':
            words1 = set(n1.split())
            words2 = set(n2.split())
            if words1 and words2:
                overlap = len(words1 & words2)
                total = len(words1 | words2)
                if overlap > 0:
                    jaccard = overlap / total
                    # High overlap is a good signal
                    if jaccard > 0.5:
                        return 0.6 + (0.4 * jaccard)

        # Check if person names match (last name + first initial)
        if entity_type == 'Person':
            parts1 = n1.split()
            parts2 = n2.split()
            if len(parts1) >= 2 and len(parts2) >= 2:
                # Check last name match
                if parts1[-1] == parts2[-1]:
                    # Check first name or initial match
                    if parts1[0] == parts2[0]:
                        return 0.95
                    elif parts1[0][0] == parts2[0][0]:
                        return 0.8

        # Fall back to sequence matching
        ratio = SequenceMatcher(None, n1, n2).ratio()
        return ratio

    @staticmethod
    def find_best_match(name: str, candidates: List, entity_type: str = None, threshold: float = 0.75):
        """Find the best matching entity from candidates."""
        best_match = None
        best_score = 0

        for candidate in candidates:
            candidate_name = candidate.canonical_name if hasattr(candidate, 'canonical_name') else str(candidate)
            score = EntityNormalizer.compute_similarity(name, candidate_name, entity_type)

            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate

        return best_match, best_score


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
        _print(f"\n{'='*60}")
        _print(f"Processing: {filepath}")
        _print('='*60)

        # Step 1: Parse document
        _print("\n[1/6] Parsing document...")
        parsed = self.parser.parse(filepath)
        if not parsed:
            _print(f"Failed to parse: {filepath}")
            return None

        # Check if already processed
        if skip_if_exists:
            existing = self.db.get_document_by_hash(parsed.file_hash)
            if existing and existing.processed_at:
                _print(f"Document already processed: {existing.id}")
                return existing.id

        # Create document record
        doc = Document.create(
            filename=parsed.filename,
            filepath=parsed.filepath,
            file_hash=parsed.file_hash
        )
        doc_id = self.db.add_document(doc)
        _print(f"Created document record: {doc_id}")

        # Step 2: Structural extraction
        _print("\n[2/6] Extracting structural elements...")
        structural = self.structural_extractor.extract(parsed.text)
        self._store_structural_results(structural, doc_id, parsed.text)

        # Step 3: Chunk document
        _print("\n[3/6] Chunking document...")
        chunks = self.chunker.chunk_text(parsed.text)
        _print(f"Created {len(chunks)} chunks")

        # Get existing entities for context
        existing_entities = [e.canonical_name for e in self.db.get_all_entities(limit=100)]

        # Step 4: Semantic extraction per chunk (PARALLEL)
        _print("\n[4/6] Extracting entities, relations, and facts (parallel)...")
        all_entities = []
        all_relations = []
        all_facts = []

        # Use parallel processing for faster extraction
        if len(chunks) > 1:
            results = self._extract_chunks_parallel(chunks, existing_entities)
        else:
            results = self._extract_chunks_sequential(chunks, existing_entities)

        for extraction, chunk in results:
            # Add span offsets to entities
            for entity in extraction.entities:
                # Ensure properties is a dict (LLM might return a list)
                if not isinstance(entity.properties, dict):
                    entity.properties = {}
                entity.properties['chunk_start'] = chunk.start_char
                entity.properties['chunk_end'] = chunk.end_char

            all_entities.extend(extraction.entities)
            all_relations.extend(extraction.relations)
            all_facts.extend(extraction.facts)

        # Infer additional relationships from extracted data
        _print("\n[4.5/6] Inferring implicit relationships...")
        inferred_relations = RelationshipInferrer.infer_relationships(
            all_entities, all_relations, all_facts
        )
        _print(f"  Inferred {len(inferred_relations)} additional relationships")
        all_relations.extend(inferred_relations)

        # Step 5: Entity resolution and storage
        _print("\n[5/6] Resolving and storing entities...")
        entity_map = self._resolve_and_store_entities(all_entities, doc_id, parsed.text)

        # Step 6: Store relations and facts
        _print("\n[6/6] Storing relations and facts...")
        self._store_relations(all_relations, entity_map, doc_id)
        self._store_facts(all_facts, entity_map, doc_id)

        # Mark document as processed
        self.db.mark_document_processed(doc_id)

        # Save vector store
        self.vector_store.save()

        _print(f"\n{'='*60}")
        _print(f"Document processing complete: {doc_id}")
        stats = self.db.get_stats()
        _print(f"Graph now has: {stats['entities']} entities, {stats['edges']} edges")
        _print('='*60)

        return doc_id

    def process_directory(self, dirpath: str, recursive: bool = True, parallel: bool = False) -> List[str]:
        """Process all documents in a directory.

        Args:
            dirpath: Path to directory containing documents
            recursive: Search subdirectories
            parallel: Process documents in parallel (experimental)

        Returns list of successfully processed document IDs.
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            _print(f"Directory not found: {dirpath}")
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

        _print(f"Found {len(files)} documents to process")

        if parallel and len(files) > 1:
            processed_ids = self._process_documents_parallel(files)
        else:
            for filepath in files:
                try:
                    doc_id = self.process_document(str(filepath))
                    if doc_id:
                        processed_ids.append(doc_id)
                except Exception as e:
                    _print(f"Error processing {filepath}: {e}")
                    continue

        return processed_ids

    def _process_documents_parallel(self, files: List[Path], max_workers: int = 2) -> List[str]:
        """Process multiple documents in parallel.

        Note: Uses limited parallelism to avoid database contention.
        """
        processed_ids = []
        total_files = len(files)

        _print(f"\nProcessing {total_files} documents with {max_workers} parallel workers...")

        def process_single(filepath: Path) -> Optional[str]:
            """Worker function for parallel document processing."""
            try:
                return self.process_document(str(filepath))
            except Exception as e:
                _print(f"Error processing {filepath}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single, f): f for f in files}

            completed = 0
            for future in as_completed(futures):
                filepath = futures[future]
                doc_id = future.result()
                completed += 1

                if doc_id:
                    processed_ids.append(doc_id)
                    _print(f"\n[{completed}/{total_files}] Completed: {filepath.name}")
                else:
                    _print(f"\n[{completed}/{total_files}] Failed: {filepath.name}")

        return processed_ids

    def _extract_chunks_parallel(
        self, chunks: List[Chunk], existing_entities: List[str]
    ) -> List[Tuple[SemanticExtraction, Chunk]]:
        """Extract from chunks in parallel using ThreadPoolExecutor."""
        results = []
        total_chunks = len(chunks)

        _print(f"  Processing {total_chunks} chunks with {MAX_PARALLEL_WORKERS} parallel workers...")

        def extract_chunk(chunk_data: Tuple[int, Chunk]) -> Tuple[SemanticExtraction, Chunk, int]:
            """Worker function for parallel extraction."""
            idx, chunk = chunk_data
            # Use global rate limiter
            _global_rate_limiter.acquire()

            try:
                extraction = self.semantic_extractor.extract(chunk.text, existing_entities)
                return extraction, chunk, idx
            except Exception as e:
                _print(f"\n    Chunk {idx+1} error: {e}")
                return SemanticExtraction(entities=[], relations=[], facts=[]), chunk, idx

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            # Submit all chunks
            futures = {
                executor.submit(extract_chunk, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                extraction, chunk, idx = future.result()
                results.append((extraction, chunk))
                completed += 1

                # Progress update
                e_count = len(extraction.entities)
                r_count = len(extraction.relations)
                f_count = len(extraction.facts)
                _print(f"  [{completed}/{total_chunks}] Chunk {idx+1}: {e_count} entities, {r_count} relations, {f_count} facts")

        # Sort results by original chunk order
        results.sort(key=lambda x: chunks.index(x[1]))

        total_e = sum(len(r[0].entities) for r in results)
        total_r = sum(len(r[0].relations) for r in results)
        total_f = sum(len(r[0].facts) for r in results)
        _print(f"  Total extracted: {total_e} entities, {total_r} relations, {total_f} facts")

        return results

    def _extract_chunks_sequential(
        self, chunks: List[Chunk], existing_entities: List[str]
    ) -> List[Tuple[SemanticExtraction, Chunk]]:
        """Extract from chunks sequentially (for single chunk or fallback)."""
        results = []

        for i, chunk in enumerate(chunks):
            _print(f"  Chunk {i+1}/{len(chunks)}...", end=" ")

            try:
                extraction = self.semantic_extractor.extract(chunk.text, existing_entities)
                results.append((extraction, chunk))

                # Update existing entities for context
                existing_entities.extend([e.name for e in extraction.entities])

                _print(f"Found {len(extraction.entities)} entities, {len(extraction.relations)} relations, {len(extraction.facts)} facts")

            except Exception as e:
                _print(f"Error: {e}")
                results.append((SemanticExtraction(entities=[], relations=[], facts=[]), chunk))

        return results

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

        _print(f"  Stored {len(structural.parties)} parties, {len(structural.defined_terms)} defined terms")

    def _resolve_and_store_entities(self, entities: List[ExtractedEntity], doc_id: str, full_text: str) -> Dict[str, str]:
        """Resolve extracted entities against existing graph and store.

        Uses EntityNormalizer for improved coreference resolution.
        Returns mapping of entity name -> entity ID.
        """
        entity_map = {}

        for entity in entities:
            if not entity.name or len(entity.name) < 2:
                continue

            # Validate and potentially correct entity type based on name patterns
            validated_type = EntityNormalizer.validate_entity_type(entity.name, entity.type)
            if validated_type != entity.type:
                entity.type = validated_type

            # Normalize the entity name
            normalized_name = EntityNormalizer.normalize_name(entity.name, entity.type)

            # Check if entity already exists (by name match)
            existing = self.db.search_entities_by_name(entity.name, limit=10)

            # Also search by normalized name if different
            if normalized_name != entity.name:
                normalized_matches = self.db.search_entities_by_name(normalized_name, limit=5)
                for nm in normalized_matches:
                    if nm not in existing:
                        existing.append(nm)

            if existing:
                # Use improved matching with EntityNormalizer
                best_match, match_score = EntityNormalizer.find_best_match(
                    entity.name, existing, entity.type, threshold=0.8
                )

                if best_match and match_score >= 0.9:
                    # High confidence match - use existing entity
                    entity_map[entity.name] = best_match.id
                    # Add as alias if name is different
                    if entity.name.lower() != best_match.canonical_name.lower():
                        self.db.add_alias(Alias.create(best_match.id, entity.name, "extracted"))
                    continue

                elif best_match and match_score >= 0.8:
                    # Good match - use existing but check embedding for confirmation
                    if self.vector_store.get_count() > 0:
                        query_embedding = self.embedding_generator.generate_query_embedding(
                            f"{entity.name} {entity.type}"
                        )
                        similar = self.vector_store.search(query_embedding, k=3)

                        # If embedding also confirms, use the match
                        for sim_id, emb_score in similar:
                            if sim_id == best_match.id and emb_score > 0.6:
                                entity_map[entity.name] = best_match.id
                                if entity.name.lower() != best_match.canonical_name.lower():
                                    self.db.add_alias(Alias.create(best_match.id, entity.name, "extracted"))
                                break
                        else:
                            # Embedding doesn't confirm - queue for review
                            self.db.add_to_resolution_queue(
                                surface_text=entity.name,
                                context=entity.span_text[:200],
                                doc_id=doc_id,
                                span_start=entity.properties.get('chunk_start', 0),
                                span_end=entity.properties.get('chunk_end', 0),
                                candidates=[{"entity_id": best_match.id, "score": float(match_score)}]
                            )
                            new_id = self._create_new_entity(entity, doc_id, full_text)
                            entity_map[entity.name] = new_id
                    else:
                        # No embeddings - trust the normalizer match
                        entity_map[entity.name] = best_match.id
                        if entity.name.lower() != best_match.canonical_name.lower():
                            self.db.add_alias(Alias.create(best_match.id, entity.name, "extracted"))
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
                                # Verify with normalizer
                                norm_score = EntityNormalizer.compute_similarity(
                                    entity.name, sim_entity.canonical_name, entity.type
                                )
                                if norm_score > 0.6 or score > 0.85:
                                    entity_map[entity.name] = sim_id
                                    self.db.add_alias(Alias.create(sim_id, entity.name, "extracted"))
                                    break
                    else:
                        # No high confidence match
                        if similar and similar[0][1] > 0.5:
                            self.db.add_to_resolution_queue(
                                surface_text=entity.name,
                                context=entity.span_text[:200],
                                doc_id=doc_id,
                                span_start=entity.properties.get('chunk_start', 0),
                                span_end=entity.properties.get('chunk_end', 0),
                                candidates=[{"entity_id": eid, "score": float(s)} for eid, s in similar[:3]]
                            )
                        new_id = self._create_new_entity(entity, doc_id, full_text)
                        entity_map[entity.name] = new_id
                else:
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
            _print(f"Warning: Could not generate embedding for {name}: {e}")

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

        _print(f"  Stored {stored} relations")

    def _store_facts(self, facts: List, entity_map: Dict[str, str], doc_id: str):
        """Store extracted facts as Fact entities."""
        stored = 0

        for fact in facts:
            try:
                # Ensure properties is a dict
                fact_props = fact.properties if isinstance(fact.properties, dict) else {}

                # Create Fact entity
                fact_entity = Entity.create(
                    type="Fact",
                    canonical_name=f"{fact.fact_type}: {fact.text[:50]}...",
                    properties={
                        "fact_type": fact.fact_type,
                        "full_text": fact.text,
                        **fact_props
                    },
                    confidence="extracted"
                )
                fact_id = self.db.add_entity(fact_entity)

                # Link to related entities
                related = fact.related_entities if isinstance(fact.related_entities, list) else []
                for entity_ref in related:
                    # Handle both string names and dict objects
                    if isinstance(entity_ref, str):
                        entity_name = entity_ref
                    elif isinstance(entity_ref, dict):
                        entity_name = entity_ref.get('name', '') or str(entity_ref)
                    else:
                        continue

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
            except Exception as e:
                _print(f"  Warning: Failed to store fact: {e}")
                continue

        _print(f"  Stored {stored} facts")

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
