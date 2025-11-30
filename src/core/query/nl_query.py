"""
Natural Language Query Engine for the Knowledge Graph.

Takes natural language questions and:
1. Interprets the query intent
2. Translates to graph operations
3. Executes the query
4. Interprets and formats results
"""
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import google.generativeai as genai

from ..storage.database import Database
from ..storage.models import Entity, Edge
from ..embeddings.vector_store import VectorStore, EmbeddingGenerator
from ..config import GEMINI_API_KEY, GEMINI_MODEL
from ..inference.graph_inference import GraphInference

# Rate limiting
MIN_DELAY_BETWEEN_REQUESTS = 4.5  # seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 45  # seconds


@dataclass
class QueryResult:
    """Result of a natural language query."""
    query: str
    interpretation: str
    entities: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    facts: List[Dict[str, Any]]
    answer: str
    confidence: float


class NLQueryEngine:
    """Natural language query engine for the knowledge graph."""

    # Cache for schema (regenerate every N queries)
    _schema_cache = None
    _schema_cache_query_count = 0
    SCHEMA_CACHE_REFRESH_INTERVAL = 50  # Refresh schema every 50 queries

    # Complex query decomposition prompt
    QUERY_DECOMPOSITION_PROMPT = """You are an expert at breaking down complex legal questions into simpler sub-queries.

Given a complex question, decompose it into 2-5 simpler, atomic questions that can be answered independently and then combined.

Complex Question: {query}

Rules:
1. Each sub-query should be answerable with a single graph operation
2. Sub-queries should be ordered logically (entities first, then relationships, then facts)
3. Include queries for both entities AND their relationships
4. For "what" questions, include queries about relevant facts
5. For comparison questions, query both sides

Output JSON array of sub-queries with their intent:
[
  {{"query": "simple question 1", "intent": "find_entities|find_relationships|find_facts|find_timeline", "depends_on": null}},
  {{"query": "simple question 2", "intent": "...", "depends_on": 0}},
  ...
]

Examples:
- "Who are the expert witnesses and what are their opinions?" ->
  [{{"query": "Who are the expert witnesses?", "intent": "find_entities", "depends_on": null}},
   {{"query": "What opinions or findings did the experts report?", "intent": "find_facts", "depends_on": 0}}]

- "What is the timeline of events between the contract signing and the dispute?" ->
  [{{"query": "When was the contract signed?", "intent": "find_timeline", "depends_on": null}},
   {{"query": "What events occurred after the contract?", "intent": "find_timeline", "depends_on": 0}},
   {{"query": "When did the dispute arise?", "intent": "find_timeline", "depends_on": null}}]

Output only the JSON array:"""

    QUERY_INTERPRETATION_PROMPT = """You are a query interpreter for a legal knowledge graph. Analyze the user's question and determine:

1. QUERY_TYPE: One of:
   - entity_search: Looking for specific entities (people, organizations, etc.)
   - relationship_query: Looking for relationships between entities
   - fact_search: Looking for specific facts, obligations, deadlines
   - path_finding: Looking for connections between entities
   - aggregation: Counting or summarizing entities/relationships
   - timeline: Looking for chronological information

2. ENTITIES_MENTIONED: Extract any entity names or types mentioned

3. RELATION_TYPES: Any relationship types implied

4. FILTERS: Any constraints (entity types, confidence levels, etc.)

5. GRAPH_OPERATIONS: List of operations to execute:
   - search_entities(name, type): Search for entities
   - get_neighbors(entity_id, hops): Get connected entities
   - find_path(entity1, entity2): Find connection between entities
   - get_facts(entity_id, fact_type): Get facts about entity
   - search_by_type(type): Get all entities of a type

User Question: {query}

=== CURRENT GRAPH SCHEMA ===
{schema}

Output JSON with the analysis:
"""

    ANSWER_GENERATION_PROMPT = """You are a legal assistant providing precise, well-cited answers from knowledge graph data extracted from legal documents.

User's Question: {query}

ENTITIES FOUND:
{entities}

RELATIONSHIPS:
{relationships}

KEY FACTS (with sources):
{facts}

Instructions for generating the answer:
1. STRUCTURE YOUR ANSWER:
   - Start with a direct answer to the question (1-2 sentences)
   - Provide supporting details with citations
   - End with any caveats or missing information

2. CITATION REQUIREMENTS:
   - Cite source documents when available (e.g., "[Source: Pre-Hearing Brief]")
   - When stating facts, indicate fact type in parentheses (allegation, finding, obligation, etc.)
   - Use entity names exactly as they appear in the data

3. FORMATTING:
   - Use bullet points for multiple items
   - Bold key names, dates, and amounts
   - Group related information under subheadings if the answer is complex
   - Use markdown formatting

4. FOR LEGAL QUESTIONS:
   - Clearly distinguish between allegations and proven facts
   - Note relevant parties, dates, monetary amounts
   - Highlight contractual obligations or deadlines
   - Identify any conflicts in the evidence

5. QUALITY CHECKS:
   - Only state what is supported by the provided data
   - If asked about something not in the data, say so clearly
   - Note if information seems incomplete or contradictory

Provide your answer:
"""

    # Enhanced fact synthesis prompt
    FACT_SYNTHESIS_PROMPT = """Based on these extracted facts from legal documents, synthesize a coherent answer.

Question: {query}

Relevant Facts:
{facts}

Related Entities:
{entities}

Synthesize these facts into a clear narrative answer that:
1. Groups related facts together
2. Identifies any contradictions or conflicts
3. Highlights key dates, amounts, and obligations
4. Notes the source/type of each fact when relevant

Synthesized Answer:
"""

    SCHEMA_EXPLORATION_PROMPT = """You are a knowledge graph query planner. The user's query returned no direct results.
Your job is to suggest alternative search strategies based on the graph schema.

User's Original Query: {query}

Graph Schema:
{schema}

Based on this schema, suggest up to 3 search strategies that might find relevant information.
For each strategy, output JSON with:
- strategy_type: one of "type_search", "keyword_search", "relationship_search", "fact_search"
- entity_types: list of entity types to search (if applicable)
- keywords: list of keywords/synonyms to search for
- relation_types: list of relation types to explore (if applicable)
- reasoning: brief explanation of why this might help

Think creatively:
- "key dates" might be found in Date entities, or in properties of other entities
- "timeline" could mean Date entities or facts with temporal info
- "money" could be Money entities or contract terms
- "obligations" are likely in Fact entities with fact_type="obligation"

Output as JSON array:
"""

    def __init__(self, db: Database, vector_store: VectorStore, api_key: str = GEMINI_API_KEY):
        self.db = db
        self.vector_store = vector_store
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.embedding_generator = EmbeddingGenerator(api_key)
        self.last_request_time = 0
        self.inference = GraphInference(db)  # Probabilistic inference engine

        self.generation_config = genai.GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=4096,
        )

    def _get_live_schema(self, force_refresh: bool = False) -> str:
        """Get live schema from the database (cached)."""
        NLQueryEngine._schema_cache_query_count += 1

        if (NLQueryEngine._schema_cache is None or
            force_refresh or
            NLQueryEngine._schema_cache_query_count >= self.SCHEMA_CACHE_REFRESH_INTERVAL):

            NLQueryEngine._schema_cache_query_count = 0

            # Get entity type counts
            stats = self.db.get_stats()
            entity_types = stats.get('entities_by_type', {})
            edge_types = stats.get('edges_by_type', {})

            # Build schema string
            schema_parts = []

            # Entity types
            schema_parts.append("ENTITY TYPES:")
            for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
                schema_parts.append(f"  - {etype}: {count} entities")

            # Top relationship types (limit to top 30)
            schema_parts.append("\nRELATIONSHIP TYPES:")
            sorted_edges = sorted(edge_types.items(), key=lambda x: -x[1])[:30]
            for rtype, count in sorted_edges:
                schema_parts.append(f"  - {rtype}: {count} edges")

            # Get sample entity names for key types
            schema_parts.append("\nKEY ENTITIES (samples):")
            for etype in ['Organization', 'Person', 'Document']:
                entities = self.db.get_entities_by_type(etype, limit=5)
                if entities:
                    names = [e.canonical_name for e in entities]
                    schema_parts.append(f"  {etype}s: {', '.join(names)}")

            # Total counts
            schema_parts.append(f"\nTOTALS: {stats.get('total_entities', 0)} entities, {stats.get('total_edges', 0)} relationships")

            NLQueryEngine._schema_cache = "\n".join(schema_parts)

        return NLQueryEngine._schema_cache

    def disambiguate_entity(self, query_name: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """
        Disambiguate a user-provided entity name to canonical entities.

        Returns a ranked list of potential matches with confidence scores.
        """
        candidates = []
        query_lower = query_name.lower().strip()

        # Search for matching entities
        matches = self.db.search_entities_by_name(query_name, limit=30)

        # Filter by type if specified
        if entity_type:
            matches = [m for m in matches if m.type == entity_type]

        for entity in matches:
            score = self._compute_entity_match_score(query_lower, entity)
            if score > 0:
                # Get aliases for context
                aliases = self.db.get_aliases(entity.id)
                alias_names = [a.alias_text for a in aliases[:5]]

                candidates.append({
                    'id': entity.id,
                    'canonical_name': entity.canonical_name,
                    'type': entity.type,
                    'confidence': score,
                    'aliases': alias_names,
                    'properties': entity.properties
                })

        # Sort by confidence
        candidates.sort(key=lambda x: -x['confidence'])
        return candidates[:10]

    def _compute_entity_match_score(self, query_lower: str, entity) -> float:
        """Compute how well an entity matches the query."""
        entity_name = entity.canonical_name.lower()
        score = 0.0

        # Exact match
        if query_lower == entity_name:
            return 1.0

        # Normalized exact match (remove common suffixes like Inc, LLC, Ltd)
        def normalize(s):
            s = s.lower()
            for suffix in [' inc', ' inc.', ' llc', ' ltd', ' ltd.', ' corp', ' corp.',
                          ' corporation', ' aerospace', ' group', ' company', ' co.']:
                s = s.replace(suffix, '')
            return s.strip()

        if normalize(query_lower) == normalize(entity_name):
            return 0.95

        # Acronym match (e.g., "CITIOM" matches "Channel IT Isle of Man")
        if len(query_lower) <= 10 and query_lower.isupper():
            words = entity_name.split()
            acronym = ''.join(w[0].upper() for w in words if w)
            if query_lower.upper() == acronym:
                return 0.9

        # Substring match
        if query_lower in entity_name:
            # Score based on how much of the name is covered
            score = len(query_lower) / len(entity_name) * 0.7
        elif entity_name in query_lower:
            score = len(entity_name) / len(query_lower) * 0.6

        # Word overlap
        query_words = set(query_lower.split())
        entity_words = set(entity_name.split())
        overlap = len(query_words & entity_words)
        if overlap > 0:
            word_score = overlap / max(len(query_words), len(entity_words)) * 0.5
            score = max(score, word_score)

        # Check aliases
        aliases = self.db.get_aliases(entity.id)
        for alias in aliases:
            alias_lower = alias.alias_text.lower()
            if query_lower == alias_lower:
                score = max(score, 0.85)
            elif query_lower in alias_lower or alias_lower in query_lower:
                score = max(score, 0.6)

        return score

    def resolve_entity_references(self, entities_mentioned: List[str]) -> List[Dict[str, Any]]:
        """
        Resolve a list of entity name references to their best canonical matches.

        For each mentioned entity, finds the best matching canonical entity.
        """
        resolved = []

        for name in entities_mentioned:
            candidates = self.disambiguate_entity(name)

            if candidates:
                best = candidates[0]
                resolved.append({
                    'mentioned': name,
                    'resolved_id': best['id'],
                    'resolved_name': best['canonical_name'],
                    'type': best['type'],
                    'confidence': best['confidence'],
                    'alternatives': candidates[1:3]  # Include next 2 alternatives
                })
            else:
                resolved.append({
                    'mentioned': name,
                    'resolved_id': None,
                    'resolved_name': None,
                    'type': None,
                    'confidence': 0.0,
                    'alternatives': []
                })

        return resolved

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
            sleep_time = MIN_DELAY_BETWEEN_REQUESTS - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _call_with_retry(self, prompt: str, config=None) -> str:
        """Call the API with retry logic for rate limits."""
        config = config or self.generation_config
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.model.generate_content(prompt, generation_config=config)
                return response.text
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower():
                    wait_time = RETRY_BASE_DELAY * (attempt + 1)
                    print(f"  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(f"Failed after {MAX_RETRIES} retries")

    def _is_complex_query(self, question: str) -> bool:
        """Determine if a query needs decomposition."""
        complex_indicators = [
            ' and ', ' or ', 'compare', 'between', 'timeline', 'history',
            'what are all', 'list all', 'summarize', 'explain the relationship',
            'how is', 'how are', 'what happened', 'sequence of events'
        ]
        q_lower = question.lower()

        # Multiple question marks or semicolons
        if question.count('?') > 1 or ';' in question:
            return True

        # Contains complex indicators
        for indicator in complex_indicators:
            if indicator in q_lower:
                return True

        # Long questions (more than 15 words) are often complex
        if len(question.split()) > 15:
            return True

        return False

    def decompose_query(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a complex query into simpler sub-queries."""
        prompt = self.QUERY_DECOMPOSITION_PROMPT.format(query=question)

        try:
            self._rate_limit()
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            response_text = response.text.strip()

            # Parse JSON response
            if response_text.startswith('```'):
                response_text = re.sub(r'```(?:json)?', '', response_text).strip()

            sub_queries = json.loads(response_text)

            if not isinstance(sub_queries, list):
                return [{"query": question, "intent": "general", "depends_on": None}]

            return sub_queries

        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [{"query": question, "intent": "general", "depends_on": None}]

    def query_complex(self, question: str) -> QueryResult:
        """Handle complex queries by decomposing and combining results."""
        # Decompose the query
        sub_queries = self.decompose_query(question)

        all_entities = []
        all_edges = []
        all_facts = []
        sub_answers = []

        # Execute each sub-query
        for i, sq in enumerate(sub_queries):
            sub_q = sq.get('query', question)
            intent = sq.get('intent', 'general')

            # Execute the sub-query
            sub_result = self._execute_single_query(sub_q)

            all_entities.extend(sub_result.entities)
            all_edges.extend(sub_result.edges)
            all_facts.extend(sub_result.facts)

            if sub_result.answer:
                sub_answers.append(f"**{sub_q}**\n{sub_result.answer}")

        # Deduplicate results
        seen_entity_ids = set()
        unique_entities = []
        for e in all_entities:
            eid = e.get('id')
            if eid and eid not in seen_entity_ids:
                seen_entity_ids.add(eid)
                unique_entities.append(e)

        seen_edge_ids = set()
        unique_edges = []
        for e in all_edges:
            eid = e.get('id')
            if eid and eid not in seen_edge_ids:
                seen_edge_ids.add(eid)
                unique_edges.append(e)

        # Generate combined answer
        if sub_answers:
            combined_answer = "\n\n".join(sub_answers)
        else:
            combined_answer = self._generate_answer(question, unique_entities, unique_edges, all_facts)

        return QueryResult(
            query=question,
            interpretation=f"Complex query decomposed into {len(sub_queries)} sub-queries",
            entities=unique_entities,
            edges=unique_edges,
            facts=all_facts,
            answer=combined_answer,
            confidence=0.8
        )

    def _execute_single_query(self, question: str) -> QueryResult:
        """Execute a single, atomic query."""
        # This calls the original query logic
        return self._query_internal(question)

    def query(self, question: str) -> QueryResult:
        """Process a natural language query and return results.

        Automatically detects complex queries and decomposes them.
        """
        # Check if this is a complex query needing decomposition
        if self._is_complex_query(question):
            print(f"\n[Complex Query Detected - Decomposing]")
            return self.query_complex(question)

        return self._query_internal(question)

    def _query_internal(self, question: str) -> QueryResult:
        """Internal query processing for single, atomic queries."""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)

        # Step 1: Interpret the query
        print("\n[1/3] Interpreting query...")
        interpretation = self._interpret_query(question)

        # Step 2: Execute graph operations
        print("\n[2/3] Executing graph operations...")
        entities, edges, facts = self._execute_operations(interpretation)

        # Step 3: Generate answer
        print("\n[3/3] Generating answer...")
        answer = self._generate_answer(question, entities, edges, facts)

        # Calculate answer confidence using Bayesian inference
        confidence_result = self.inference.score_answer_confidence(entities, facts, edges)
        confidence = confidence_result['confidence']

        return QueryResult(
            query=question,
            interpretation=interpretation.get('query_type', 'unknown'),
            entities=[e.to_dict() if hasattr(e, 'to_dict') else e for e in entities],
            edges=[e.to_dict() if hasattr(e, 'to_dict') else e for e in edges],
            facts=[f for f in facts],
            answer=answer,
            confidence=confidence
        )

    def _interpret_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to interpret the query and determine operations."""
        # Always get fallback interpretation first
        fallback = self._fallback_interpretation(query)

        # Get live schema for context
        schema = self._get_live_schema()

        prompt = self.QUERY_INTERPRETATION_PROMPT.format(query=query, schema=schema)

        try:
            response_text = self._call_with_retry(prompt)

            # Parse JSON response
            text = response_text.strip()

            # Remove markdown code blocks
            if text.startswith('```'):
                lines = text.split('\n')
                start = 1 if lines[0].startswith('```') else 0
                end = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '```':
                        end = i
                        break
                text = '\n'.join(lines[start:end])

            interpretation = json.loads(text)

            # Merge with fallback - use fallback values if LLM didn't provide them
            query_type = interpretation.get('query_type', '')
            if query_type in ['', 'unknown'] and fallback.get('query_type', 'unknown') != 'entity_search':
                interpretation['query_type'] = fallback['query_type']

            # Add entity types from fallback if not in LLM response
            if 'entity_types_requested' not in interpretation:
                interpretation['entity_types_requested'] = fallback.get('entity_types_requested', [])

            # Merge entities from fallback
            llm_entities = interpretation.get('entities_mentioned', [])
            fallback_entities = fallback.get('entities_mentioned', [])
            if not llm_entities:
                interpretation['entities_mentioned'] = fallback_entities
            else:
                # Add any fallback entities not in LLM list
                for e in fallback_entities:
                    if e not in llm_entities:
                        llm_entities.append(e)

            # Store original query for fallback exploration
            interpretation['_original_query'] = query

            print(f"  Query type: {interpretation.get('query_type', 'unknown')}")
            print(f"  Entities: {interpretation.get('entities_mentioned', [])}")
            print(f"  Entity types: {interpretation.get('entity_types_requested', [])}")
            return interpretation

        except Exception as e:
            print(f"  Error interpreting query: {e}")
            # Use fallback interpretation
            fallback['_original_query'] = query
            print(f"  Using fallback - Query type: {fallback.get('query_type', 'unknown')}")
            return fallback

    def _fallback_interpretation(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based query interpretation as fallback."""
        query_lower = query.lower()

        interpretation = {
            "query_type": "entity_search",
            "entities_mentioned": [],
            "relation_types": [],
            "filters": {},
            "graph_operations": [],
            "entity_types_requested": []
        }

        # Detect query type
        if any(w in query_lower for w in ['relationship', 'related', 'connected', 'between']):
            interpretation['query_type'] = 'relationship_query'
        elif any(w in query_lower for w in ['obligation', 'deadline', 'must', 'shall']):
            interpretation['query_type'] = 'fact_search'
        elif any(w in query_lower for w in ['path', 'connection', 'link']):
            interpretation['query_type'] = 'path_finding'
        elif any(w in query_lower for w in ['how many', 'count', 'all']):
            interpretation['query_type'] = 'aggregation'
        elif any(w in query_lower for w in ['parties', 'party', 'plaintiff', 'defendant', 'claimant', 'respondent']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Organization', 'Person']
        elif any(w in query_lower for w in ['person', 'people', 'who', 'witness', 'witnesses']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Person']
        elif any(w in query_lower for w in ['company', 'companies', 'organization', 'corporation']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Organization']
        elif any(w in query_lower for w in ['allegation', 'allegations', 'claim', 'claims', 'allege']):
            interpretation['query_type'] = 'fact_search'
            interpretation['filters']['fact_type'] = 'allegation'
        # NEW: Date/timeline patterns
        elif any(w in query_lower for w in ['date', 'dates', 'when', 'timeline', 'chronolog', 'time']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Date']
        # NEW: Money/financial patterns
        elif any(w in query_lower for w in ['money', 'amount', 'dollar', 'payment', 'sum', 'damages', 'cost', 'price', 'value']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Money']
        # NEW: Location patterns
        elif any(w in query_lower for w in ['location', 'where', 'place', 'address', 'city', 'state', 'country']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Location']
        # NEW: Document patterns
        elif any(w in query_lower for w in ['document', 'contract', 'agreement', 'exhibit', 'filing', 'motion']):
            interpretation['query_type'] = 'entity_search'
            interpretation['entity_types_requested'] = ['Document']
        # NEW: Fact/information patterns
        elif any(w in query_lower for w in ['fact', 'facts', 'information', 'detail', 'details']):
            interpretation['query_type'] = 'fact_search'
        # NEW: General overview/summary queries - catch "dispute", "case", "about", etc.
        elif any(w in query_lower for w in ['dispute', 'case', 'lawsuit', 'litigation', 'matter', 'summary', 'summarize', 'overview', 'about']):
            interpretation['query_type'] = 'overview'
            interpretation['entity_types_requested'] = ['Organization', 'Person', 'Fact']

        # Extract quoted entities
        quoted = re.findall(r'"([^"]+)"', query)
        interpretation['entities_mentioned'] = quoted

        # Extract capitalized words as potential entities
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        for cap in caps:
            if cap not in interpretation['entities_mentioned'] and cap.lower() not in ['who', 'what', 'where', 'when', 'how', 'why', 'the']:
                interpretation['entities_mentioned'].append(cap)

        return interpretation

    def _execute_operations(self, interpretation: Dict[str, Any]) -> Tuple[List[Entity], List[Edge], List[Dict]]:
        """Execute graph operations based on interpretation."""
        entities = []
        edges = []
        facts = []
        entity_ids_seen = set()

        query_type = interpretation.get('query_type', 'entity_search')
        mentioned = interpretation.get('entities_mentioned', [])
        operations = interpretation.get('graph_operations', [])
        entity_types_requested = interpretation.get('entity_types_requested', [])

        # Use disambiguation to find mentioned entities
        for name in mentioned:
            if isinstance(name, str):
                # Use disambiguation for better matching
                candidates = self.disambiguate_entity(name)
                for cand in candidates[:3]:  # Top 3 matches for each mention
                    if cand['confidence'] > 0.3 and cand['id'] not in entity_ids_seen:
                        entity = self.db.get_entity(cand['id'])
                        if entity:
                            entities.append(entity)
                            entity_ids_seen.add(cand['id'])

                # Fall back to simple search if no disambiguation results
                if not candidates:
                    found = self.db.search_entities_by_name(name, limit=10)
                    for entity in found:
                        if entity.id not in entity_ids_seen:
                            entities.append(entity)
                            entity_ids_seen.add(entity.id)

        # If no entities found but specific types requested, get entities by type
        if not entities and entity_types_requested:
            for entity_type in entity_types_requested:
                type_entities = self.db.get_entities_by_type(entity_type, limit=20)
                entities.extend(type_entities)

        # Role-based person search: look for facts mentioning roles
        query_lower = interpretation.get('_original_query', '').lower()
        role_keywords = {
            'expert': ['expert', 'expert witness', 'damages expert', 'rebuttal expert'],
            'witness': ['witness', 'expert witness', 'testif'],
            'counsel': ['counsel', 'attorney', 'lawyer', 'law firm'],
            'arbitrator': ['arbitrator', 'arbitral', 'panel'],
        }

        for role, keywords in role_keywords.items():
            if role in query_lower:
                # Search for facts containing role keywords directly
                # Use text search on entity names which include "expert", etc.
                for kw in keywords:
                    matched_entities = self.db.search_entities_by_name(kw, limit=50)
                    for entity in matched_entities:
                        if entity.type == "Fact" and entity.id not in entity_ids_seen:
                            facts.append({
                                "type": entity.properties.get("fact_type", "finding"),
                                "text": entity.properties.get("full_text", entity.canonical_name),
                                "entity_name": entity.canonical_name,
                                "role_keyword": role
                            })
                            entities.append(entity)
                            entity_ids_seen.add(entity.id)

                            # Try to extract person names from this fact
                            full_text = entity.properties.get('full_text', entity.canonical_name)
                            # Search for people entities mentioned in this fact
                            people = self.db.get_entities_by_type("Person", limit=100)
                            for person in people:
                                if person.canonical_name.lower() in full_text.lower():
                                    if person.id not in entity_ids_seen:
                                        entities.append(person)
                                        entity_ids_seen.add(person.id)
                break  # Only match one role keyword group

        # If no entities found from names, try embedding search
        if not entities and mentioned:
            for name in mentioned:
                if isinstance(name, str):
                    try:
                        query_emb = self.embedding_generator.generate_query_embedding(name)
                        similar = self.vector_store.search(query_emb, k=5)
                        for entity_id, score in similar:
                            if score > 0.5:
                                entity = self.db.get_entity(entity_id)
                                if entity:
                                    entities.append(entity)
                    except Exception as e:
                        print(f"  Embedding search error: {e}")

        # Execute based on query type
        if query_type == 'relationship_query' and len(entities) >= 1:
            # Get relationships for found entities
            for entity in entities[:5]:  # Limit to avoid explosion
                outgoing = self.db.get_edges_from(entity.id)
                incoming = self.db.get_edges_to(entity.id)
                edges.extend(outgoing)
                edges.extend(incoming)

        elif query_type == 'fact_search':
            # Search for Fact entities
            fact_entities = self.db.get_entities_by_type("Fact", limit=50)

            # Filter by mentioned entities if any
            for fact in fact_entities:
                fact_dict = {
                    "type": fact.properties.get("fact_type", "fact"),
                    "text": fact.properties.get("full_text", fact.canonical_name),
                    "entity_name": fact.canonical_name
                }
                facts.append(fact_dict)

        elif query_type == 'path_finding' and len(entities) >= 2:
            # Find paths between entities
            path_entities, path_edges = self._find_path(entities[0].id, entities[1].id)
            entities.extend(path_entities)
            edges.extend(path_edges)

        elif query_type == 'aggregation':
            # Smart aggregation: if asking about specific things, search for them
            query_lower = interpretation.get('_original_query', '').lower()

            # Check if asking about specific fact types
            fact_type_keywords = {
                'allegation': 'allegation',
                'allegations': 'allegation',
                'finding': 'finding',
                'findings': 'finding',
                'obligation': 'obligation',
                'obligations': 'obligation',
                'deadline': 'deadline',
                'deadlines': 'deadline',
                'claim': 'allegation',
                'claims': 'allegation',
            }

            target_fact_type = None
            for keyword, fact_type in fact_type_keywords.items():
                if keyword in query_lower:
                    target_fact_type = fact_type
                    break

            if target_fact_type:
                # Search for facts of this type
                all_facts = self.db.get_entities_by_type("Fact", limit=200)
                matching_facts = []
                for fact in all_facts:
                    fact_name_lower = fact.canonical_name.lower()
                    if fact_name_lower.startswith(target_fact_type + ':') or \
                       fact.properties.get('fact_type', '').lower() == target_fact_type:
                        matching_facts.append(fact)

                # If entities mentioned, filter to facts related to those entities
                mentioned_entities = interpretation.get('entities_mentioned', [])
                if mentioned_entities:
                    # Get entity IDs for mentioned entities
                    mentioned_entity_objs = []
                    for name in mentioned_entities:
                        found = self.db.search_entities_by_name(name, limit=5)
                        mentioned_entity_objs.extend(found)

                    if mentioned_entity_objs:
                        # Filter facts to those connected to mentioned entities
                        filtered_facts = []
                        for fact in matching_facts:
                            # Check if fact text mentions any of the entity names
                            fact_text = (fact.canonical_name + ' ' +
                                       fact.properties.get('full_text', '')).lower()
                            for ent in mentioned_entity_objs:
                                if ent.canonical_name.lower() in fact_text:
                                    filtered_facts.append(fact)
                                    break
                        if filtered_facts:
                            matching_facts = filtered_facts

                # Add matching facts to results
                for fact in matching_facts[:50]:
                    facts.append({
                        "type": fact.properties.get("fact_type", target_fact_type),
                        "text": fact.properties.get("full_text", fact.canonical_name),
                        "entity_name": fact.canonical_name
                    })
                    entities.append(fact)
            else:
                # Fallback to basic stats
                stats = self.db.get_stats()
                facts.append({
                    "type": "aggregation",
                    "text": f"Graph contains: {stats['entities']} entities, {stats['edges']} edges, {stats['documents']} documents",
                    "details": stats
                })

        elif query_type == 'overview':
            # Get main parties - organizations and people
            orgs = self.db.get_entities_by_type("Organization", limit=15)
            people = self.db.get_entities_by_type("Person", limit=15)
            entities.extend(orgs)
            entities.extend(people)

            # Get key facts to understand what the dispute/case is about
            fact_entities = self.db.get_entities_by_type("Fact", limit=30)
            for fact in fact_entities:
                facts.append({
                    "type": fact.properties.get("fact_type", "fact"),
                    "text": fact.properties.get("full_text", fact.canonical_name),
                    "entity_name": fact.canonical_name
                })
                entities.append(fact)

            # Also get key relationships between parties
            for entity in orgs[:5] + people[:5]:
                outgoing = self.db.get_edges_from(entity.id)
                incoming = self.db.get_edges_to(entity.id)
                edges.extend(outgoing[:5])
                edges.extend(incoming[:5])

        # Always get neighbors for found entities (skip for aggregation/overview which already handle this)
        if entities and query_type not in ['aggregation', 'overview']:
            for entity in entities[:3]:
                neighbors = self.db.get_entity_neighbors(entity.id, max_hops=1)
                for e in neighbors['entities']:
                    if e.id not in [x.id for x in entities]:
                        entities.append(e)
                edges.extend(neighbors['edges'])

        # Deduplicate
        seen_entity_ids = set()
        unique_entities = []
        for e in entities:
            if e.id not in seen_entity_ids:
                seen_entity_ids.add(e.id)
                unique_entities.append(e)

        seen_edge_ids = set()
        unique_edges = []
        for e in edges:
            if e.id not in seen_edge_ids:
                seen_edge_ids.add(e.id)
                unique_edges.append(e)

        print(f"  Found {len(unique_entities)} entities, {len(unique_edges)} edges, {len(facts)} facts")

        # If no results found, use schema-aware exploration
        if not unique_entities and not unique_edges and not facts:
            return self._explore_with_schema(interpretation.get('_original_query', ''), interpretation)

        return unique_entities, unique_edges, facts

    def _find_path(self, start_id: str, end_id: str, max_depth: int = 3) -> Tuple[List[Entity], List[Edge]]:
        """Find a path between two entities using BFS."""
        from collections import deque

        visited = set()
        queue = deque([(start_id, [], [])])  # (current_id, path_entities, path_edges)

        while queue:
            current_id, path_entities, path_edges = queue.popleft()

            if current_id == end_id:
                # Found path
                entities = []
                for eid in path_entities + [end_id]:
                    e = self.db.get_entity(eid)
                    if e:
                        entities.append(e)
                return entities, path_edges

            if current_id in visited or len(path_entities) >= max_depth:
                continue

            visited.add(current_id)

            # Get neighbors
            outgoing = self.db.get_edges_from(current_id)
            incoming = self.db.get_edges_to(current_id)

            for edge in outgoing:
                if edge.target_entity_id not in visited:
                    queue.append((
                        edge.target_entity_id,
                        path_entities + [current_id],
                        path_edges + [edge]
                    ))

            for edge in incoming:
                if edge.source_entity_id not in visited:
                    queue.append((
                        edge.source_entity_id,
                        path_entities + [current_id],
                        path_edges + [edge]
                    ))

        return [], []  # No path found

    def _multi_hop_explore(self, start_entities: List[Entity], max_hops: int = 2,
                           relation_filter: List[str] = None) -> Tuple[List[Entity], List[Edge]]:
        """Explore graph from starting entities up to max_hops away.

        This enables multi-hop reasoning - finding connections that span multiple relationships.
        """
        all_entities = list(start_entities)
        all_edges = []
        visited_entity_ids = {e.id for e in start_entities}
        current_frontier = [e.id for e in start_entities]

        for hop in range(max_hops):
            next_frontier = []

            for entity_id in current_frontier:
                # Get outgoing edges
                outgoing = self.db.get_edges_from(entity_id)
                for edge in outgoing:
                    # Apply relation filter if specified
                    if relation_filter and edge.relation_type not in relation_filter:
                        continue

                    if edge.target_entity_id not in visited_entity_ids:
                        target = self.db.get_entity(edge.target_entity_id)
                        if target:
                            all_entities.append(target)
                            visited_entity_ids.add(edge.target_entity_id)
                            next_frontier.append(edge.target_entity_id)
                    all_edges.append(edge)

                # Get incoming edges
                incoming = self.db.get_edges_to(entity_id)
                for edge in incoming:
                    if relation_filter and edge.relation_type not in relation_filter:
                        continue

                    if edge.source_entity_id not in visited_entity_ids:
                        source = self.db.get_entity(edge.source_entity_id)
                        if source:
                            all_entities.append(source)
                            visited_entity_ids.add(edge.source_entity_id)
                            next_frontier.append(edge.source_entity_id)
                    all_edges.append(edge)

            current_frontier = next_frontier
            if not current_frontier:
                break

        return all_entities, all_edges

    def _find_all_paths(self, start_id: str, end_id: str, max_depth: int = 4) -> List[Tuple[List[str], List[Edge]]]:
        """Find all paths between two entities (up to max_depth)."""
        all_paths = []

        def dfs(current_id: str, target_id: str, path: List[str], edges: List[Edge], visited: set):
            if len(path) > max_depth:
                return

            if current_id == target_id:
                all_paths.append((path.copy(), edges.copy()))
                return

            visited.add(current_id)

            # Outgoing edges
            for edge in self.db.get_edges_from(current_id):
                neighbor = edge.target_entity_id
                if neighbor not in visited:
                    path.append(neighbor)
                    edges.append(edge)
                    dfs(neighbor, target_id, path, edges, visited)
                    path.pop()
                    edges.pop()

            # Incoming edges
            for edge in self.db.get_edges_to(current_id):
                neighbor = edge.source_entity_id
                if neighbor not in visited:
                    path.append(neighbor)
                    edges.append(edge)
                    dfs(neighbor, target_id, path, edges, visited)
                    path.pop()
                    edges.pop()

            visited.remove(current_id)

        dfs(start_id, end_id, [start_id], [], set())
        return all_paths

    def find_connections(self, entity_name1: str, entity_name2: str) -> Dict[str, Any]:
        """Find all connections between two named entities.

        Returns structured data about how the entities are connected.
        """
        # Find the entities
        e1_matches = self.db.search_entities_by_name(entity_name1, limit=3)
        e2_matches = self.db.search_entities_by_name(entity_name2, limit=3)

        if not e1_matches or not e2_matches:
            return {
                'found': False,
                'message': f"Could not find entities matching '{entity_name1}' and/or '{entity_name2}'"
            }

        e1 = e1_matches[0]
        e2 = e2_matches[0]

        # Find all paths
        all_paths = self._find_all_paths(e1.id, e2.id, max_depth=4)

        if not all_paths:
            return {
                'found': False,
                'entity1': {'id': e1.id, 'name': e1.canonical_name, 'type': e1.type},
                'entity2': {'id': e2.id, 'name': e2.canonical_name, 'type': e2.type},
                'message': 'No connection found within 4 hops'
            }

        # Format paths
        formatted_paths = []
        for path_ids, edges in all_paths[:10]:  # Limit to 10 paths
            path_entities = []
            for eid in path_ids:
                entity = self.db.get_entity(eid)
                if entity:
                    path_entities.append({
                        'id': eid,
                        'name': entity.canonical_name,
                        'type': entity.type
                    })

            path_edges = []
            for edge in edges:
                path_edges.append({
                    'relation': edge.relation_type,
                    'source': edge.source_entity_id,
                    'target': edge.target_entity_id
                })

            formatted_paths.append({
                'length': len(path_ids) - 1,
                'entities': path_entities,
                'edges': path_edges
            })

        # Sort by path length
        formatted_paths.sort(key=lambda p: p['length'])

        return {
            'found': True,
            'entity1': {'id': e1.id, 'name': e1.canonical_name, 'type': e1.type},
            'entity2': {'id': e2.id, 'name': e2.canonical_name, 'type': e2.type},
            'num_paths': len(all_paths),
            'shortest_path_length': formatted_paths[0]['length'] if formatted_paths else None,
            'paths': formatted_paths
        }

    def _get_graph_schema(self) -> str:
        """Get a description of the graph schema for LLM exploration."""
        stats = self.db.get_stats()

        # Get entity type counts
        entity_types = {}
        for entity_type in ['Person', 'Organization', 'Document', 'Date', 'Money', 'Location', 'Reference', 'Fact', 'Clause']:
            entities = self.db.get_entities_by_type(entity_type, limit=1)
            count = len(self.db.get_entities_by_type(entity_type, limit=1000))
            if count > 0:
                entity_types[entity_type] = count

        # Get relation types from edges
        relation_types = set()
        all_edges = self.db.get_all_edges(limit=500)
        for edge in all_edges:
            relation_types.add(edge.relation_type)

        # Get sample entities for context
        sample_entities = []
        for entity_type in ['Organization', 'Person', 'Date', 'Money']:
            samples = self.db.get_entities_by_type(entity_type, limit=3)
            for s in samples:
                sample_entities.append(f"  - {s.canonical_name} ({entity_type})")

        schema = f"""
GRAPH STATISTICS:
- Total entities: {stats.get('entities', 0)}
- Total relationships: {stats.get('edges', 0)}
- Total documents: {stats.get('documents', 0)}

ENTITY TYPES AND COUNTS:
{chr(10).join(f'  - {t}: {c} entities' for t, c in sorted(entity_types.items(), key=lambda x: -x[1]))}

RELATIONSHIP TYPES:
{chr(10).join(f'  - {r}' for r in sorted(relation_types))}

SAMPLE ENTITIES:
{chr(10).join(sample_entities[:12])}
"""
        return schema

    def _explore_with_schema(self, query: str, interpretation: Dict[str, Any]) -> Tuple[List[Entity], List[Edge], List[Dict]]:
        """Use LLM to explore the graph when direct search returns nothing."""
        print("  No direct results. Using schema-aware exploration...")

        entities = []
        edges = []
        facts = []

        # Get schema
        schema = self._get_graph_schema()

        # Ask LLM for search strategies
        prompt = self.SCHEMA_EXPLORATION_PROMPT.format(query=query, schema=schema)

        try:
            response_text = self._call_with_retry(prompt)

            # Parse strategies
            text = response_text.strip()
            if text.startswith('```'):
                lines = text.split('\n')
                start = 1 if lines[0].startswith('```') else 0
                end = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '```':
                        end = i
                        break
                text = '\n'.join(lines[start:end])

            strategies = json.loads(text)
            print(f"  LLM suggested {len(strategies)} search strategies")

            # Execute each strategy
            for strategy in strategies[:3]:  # Limit to 3 strategies
                strategy_type = strategy.get('strategy_type', '')
                print(f"    Trying: {strategy.get('reasoning', strategy_type)[:60]}...")

                if strategy_type == 'type_search':
                    for entity_type in strategy.get('entity_types', []):
                        type_entities = self.db.get_entities_by_type(entity_type, limit=20)
                        entities.extend(type_entities)

                elif strategy_type == 'keyword_search':
                    for keyword in strategy.get('keywords', []):
                        found = self.db.search_entities_by_name(keyword, limit=10)
                        entities.extend(found)

                        # Also try embedding search for semantic matching
                        try:
                            query_emb = self.embedding_generator.generate_query_embedding(keyword)
                            similar = self.vector_store.search(query_emb, k=5)
                            for entity_id, score in similar:
                                if score > 0.4:  # Lower threshold for exploration
                                    entity = self.db.get_entity(entity_id)
                                    if entity:
                                        entities.append(entity)
                        except Exception:
                            pass

                elif strategy_type == 'relationship_search':
                    for rel_type in strategy.get('relation_types', []):
                        # Get edges of this type
                        all_edges = self.db.get_all_edges(limit=200)
                        for edge in all_edges:
                            if edge.relation_type == rel_type:
                                edges.append(edge)
                                # Get connected entities
                                source = self.db.get_entity(edge.source_entity_id)
                                target = self.db.get_entity(edge.target_entity_id)
                                if source:
                                    entities.append(source)
                                if target:
                                    entities.append(target)

                elif strategy_type == 'fact_search':
                    fact_entities = self.db.get_entities_by_type("Fact", limit=50)
                    for fact in fact_entities:
                        # Check if fact matches any keywords
                        fact_text = fact.properties.get("full_text", fact.canonical_name).lower()
                        keywords = strategy.get('keywords', [])
                        if any(kw.lower() in fact_text for kw in keywords) or not keywords:
                            facts.append({
                                "type": fact.properties.get("fact_type", "fact"),
                                "text": fact.properties.get("full_text", fact.canonical_name),
                                "entity_name": fact.canonical_name
                            })
                            entities.append(fact)

        except Exception as e:
            print(f"  Schema exploration error: {e}")
            # Fallback: just get some entities of common types
            for entity_type in ['Date', 'Money', 'Organization', 'Person']:
                type_entities = self.db.get_entities_by_type(entity_type, limit=10)
                entities.extend(type_entities)

        # Deduplicate
        seen_ids = set()
        unique_entities = []
        for e in entities:
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                unique_entities.append(e)

        seen_edge_ids = set()
        unique_edges = []
        for e in edges:
            if e.id not in seen_edge_ids:
                seen_edge_ids.add(e.id)
                unique_edges.append(e)

        print(f"  Exploration found {len(unique_entities)} entities, {len(unique_edges)} edges, {len(facts)} facts")

        return unique_entities, unique_edges, facts

    def _generate_answer(self, query: str, entities: List[Entity], edges: List[Edge], facts: List[Dict]) -> str:
        """Generate natural language answer from graph data."""
        # Format entities
        entities_str = ""
        if entities:
            entity_lines = []
            for e in entities[:25]:
                props = e.properties if hasattr(e, 'properties') else {}
                line = f"- {e.canonical_name} ({e.type})"
                if props.get('role'):
                    line += f" - Role: {props['role']}"
                entity_lines.append(line)
            entities_str = "\n".join(entity_lines)
        else:
            entities_str = "None found"

        # Format relationships
        relationships_str = ""
        if edges:
            rel_lines = []
            for edge in edges[:25]:
                source = self.db.get_entity(edge.source_entity_id)
                target = self.db.get_entity(edge.target_entity_id)
                source_name = source.canonical_name if source else "Unknown"
                target_name = target.canonical_name if target else "Unknown"
                rel_lines.append(f"- {source_name} --[{edge.relation_type}]--> {target_name}")
            relationships_str = "\n".join(rel_lines)
        else:
            relationships_str = "None found"

        # Format facts with source citations
        facts_str = ""
        if facts:
            fact_lines = []
            for f in facts[:20]:  # Include more facts
                fact_type = f.get('type', f.get('fact_type', 'fact'))
                text = f.get('text', f.get('full_text', ''))[:300]
                source = f.get('source_doc', f.get('provenance_doc_id', ''))
                confidence = f.get('confidence', '')

                line = f"- [{fact_type.upper()}] {text}"
                if source:
                    line += f" [Source: {source[:30]}]"
                if confidence:
                    line += f" (confidence: {confidence})"
                fact_lines.append(line)
            facts_str = "\n".join(fact_lines)
        else:
            facts_str = "None found"

        # Also include Fact entities as additional context
        fact_entities = [e for e in entities if e.type == 'Fact']
        if fact_entities and len(facts_str) < 2000:
            facts_str += "\n\nADDITIONAL FACTS FROM ENTITIES:"
            for fe in fact_entities[:10]:
                fact_text = fe.properties.get('full_text', fe.canonical_name)[:200]
                fact_type = fe.properties.get('fact_type', 'fact')
                facts_str += f"\n- [{fact_type.upper()}] {fact_text}"

        # Generate answer with structured prompt
        prompt = self.ANSWER_GENERATION_PROMPT.format(
            query=query,
            entities=entities_str,
            relationships=relationships_str,
            facts=facts_str
        )

        try:
            answer_config = genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            )
            response_text = self._call_with_retry(prompt, answer_config)
            return response_text.strip()
        except Exception as e:
            print(f"  Error generating answer: {e}")
            return f"Found {len(entities)} entities and {len(edges)} relationships, but could not generate a detailed answer."

    def get_entity_summary(self, entity_name: str) -> str:
        """Get a summary of everything known about an entity."""
        # Find entity
        entities = self.db.search_entities_by_name(entity_name, limit=1)
        if not entities:
            return f"No entity found matching '{entity_name}'"

        entity = entities[0]

        # Get all related information
        outgoing = self.db.get_edges_from(entity.id)
        incoming = self.db.get_edges_to(entity.id)
        mentions = self.db.get_mentions_for_entity(entity.id)
        aliases = self.db.get_aliases(entity.id)

        # Build summary
        summary_parts = [
            f"## {entity.canonical_name}",
            f"**Type:** {entity.type}",
            f"**Confidence:** {entity.confidence}",
        ]

        if entity.properties:
            summary_parts.append("\n**Properties:**")
            for k, v in entity.properties.items():
                summary_parts.append(f"  - {k}: {v}")

        if aliases:
            summary_parts.append(f"\n**Also known as:** {', '.join(a.alias_text for a in aliases)}")

        if outgoing:
            summary_parts.append("\n**Relationships (outgoing):**")
            for edge in outgoing[:10]:
                target = self.db.get_entity(edge.target_entity_id)
                target_name = target.canonical_name if target else "Unknown"
                summary_parts.append(f"  - {edge.relation_type} → {target_name}")

        if incoming:
            summary_parts.append("\n**Relationships (incoming):**")
            for edge in incoming[:10]:
                source = self.db.get_entity(edge.source_entity_id)
                source_name = source.canonical_name if source else "Unknown"
                summary_parts.append(f"  - {source_name} → {edge.relation_type}")

        if mentions:
            summary_parts.append(f"\n**Mentioned in:** {len(mentions)} locations")

        return "\n".join(summary_parts)

    def list_entities(self, entity_type: str = None, limit: int = 50) -> List[Dict]:
        """List entities, optionally filtered by type."""
        if entity_type:
            entities = self.db.get_entities_by_type(entity_type, limit=limit)
        else:
            entities = self.db.get_all_entities(limit=limit)

        return [
            {
                "id": e.id,
                "name": e.canonical_name,
                "type": e.type,
                "confidence": e.confidence
            }
            for e in entities
        ]

    def _parse_date(self, date_str: str) -> Optional[Tuple[int, int, int]]:
        """Parse date string into (year, month, day) tuple."""
        import re
        from datetime import datetime

        if not date_str:
            return None

        # Try common date formats
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y',
            '%Y', '%B %Y', '%b %Y', '%m/%Y', '%Y/%m/%d'
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return (dt.year, dt.month, dt.day)
            except ValueError:
                continue

        # Try to extract year at least
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return (int(year_match.group()), 1, 1)

        return None

    def _extract_date_from_entity(self, entity) -> Optional[Tuple[int, int, int]]:
        """Extract date from Date entity or entity properties."""
        if entity.type == 'Date':
            # Try canonical name first
            parsed = self._parse_date(entity.canonical_name)
            if parsed:
                return parsed

            # Try properties
            props = entity.properties or {}
            for key in ['date', 'value', 'full_date']:
                if key in props:
                    parsed = self._parse_date(str(props[key]))
                    if parsed:
                        return parsed

        # Check properties for date fields on any entity
        props = entity.properties or {}
        for key in ['date', 'due_date', 'deadline', 'effective_date', 'filing_date']:
            if key in props:
                parsed = self._parse_date(str(props[key]))
                if parsed:
                    return parsed

        return None

    def temporal_query(self, start_year: int = None, end_year: int = None,
                      start_month: int = None, end_month: int = None) -> Dict[str, Any]:
        """Query entities and facts within a time range.

        Returns entities and facts organized chronologically.
        """
        results = {
            'dates': [],
            'facts_with_dates': [],
            'timeline': []
        }

        # Get all Date entities
        date_entities = self.db.get_entities_by_type('Date', limit=500)

        for entity in date_entities:
            parsed = self._extract_date_from_entity(entity)
            if not parsed:
                continue

            year, month, day = parsed

            # Apply filters
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
            if start_month and month < start_month:
                continue
            if end_month and month > end_month:
                continue

            # Get related entities and facts
            neighbors = self.db.get_entity_neighbors(entity.id, max_hops=1)

            related_facts = []
            related_entities = []
            for e in neighbors['entities']:
                if e.type == 'Fact':
                    related_facts.append({
                        'id': e.id,
                        'text': e.properties.get('full_text', e.canonical_name)[:200],
                        'type': e.properties.get('fact_type', 'fact')
                    })
                else:
                    related_entities.append({
                        'id': e.id,
                        'name': e.canonical_name,
                        'type': e.type
                    })

            results['dates'].append({
                'id': entity.id,
                'name': entity.canonical_name,
                'date': {'year': year, 'month': month, 'day': day},
                'properties': entity.properties,
                'related_facts': related_facts,
                'related_entities': related_entities
            })

        # Get Facts with date properties
        fact_entities = self.db.get_entities_by_type('Fact', limit=500)
        for fact in fact_entities:
            props = fact.properties or {}
            for date_key in ['due_date', 'deadline', 'date', 'effective_date']:
                if date_key in props:
                    parsed = self._parse_date(str(props[date_key]))
                    if parsed:
                        year, month, day = parsed
                        if start_year and year < start_year:
                            continue
                        if end_year and year > end_year:
                            continue

                        results['facts_with_dates'].append({
                            'id': fact.id,
                            'text': props.get('full_text', fact.canonical_name)[:200],
                            'fact_type': props.get('fact_type', 'fact'),
                            'date': {'year': year, 'month': month, 'day': day},
                            'date_field': date_key
                        })
                        break

        # Build unified timeline
        timeline = []

        for d in results['dates']:
            timeline.append({
                'date': d['date'],
                'type': 'date_entity',
                'name': d['name'],
                'id': d['id'],
                'related_count': len(d['related_facts']) + len(d['related_entities'])
            })

        for f in results['facts_with_dates']:
            timeline.append({
                'date': f['date'],
                'type': 'fact',
                'name': f['text'][:100],
                'id': f['id'],
                'fact_type': f['fact_type']
            })

        # Sort by date
        def sort_key(item):
            d = item['date']
            return (d['year'], d['month'], d['day'])

        timeline.sort(key=sort_key)
        results['timeline'] = timeline

        return results

    def query_by_timeframe(self, query: str) -> Dict[str, Any]:
        """Parse temporal query and return results.

        Supports queries like:
        - "What happened in 2023?"
        - "Show events before January 2024"
        - "Timeline from 2022 to 2023"
        """
        import re

        query_lower = query.lower()

        # Extract year ranges
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        years = [int(y[:4]) if len(y) >= 4 else int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', query)]

        start_year = None
        end_year = None

        if 'before' in query_lower and years:
            end_year = years[0]
        elif 'after' in query_lower and years:
            start_year = years[0]
        elif 'from' in query_lower and 'to' in query_lower and len(years) >= 2:
            start_year = min(years)
            end_year = max(years)
        elif 'between' in query_lower and len(years) >= 2:
            start_year = min(years)
            end_year = max(years)
        elif years:
            # Single year - treat as that year only
            start_year = years[0]
            end_year = years[0]

        return self.temporal_query(start_year=start_year, end_year=end_year)

    def generate_narrative_timeline(self, start_year: int = None, end_year: int = None) -> Dict[str, Any]:
        """
        Generate a lawyer-friendly narrative timeline of the case.

        Returns structured timeline with events and an LLM-generated narrative summary.
        """
        # Get temporal data
        temporal_data = self.temporal_query(start_year=start_year, end_year=end_year)

        if not temporal_data['timeline']:
            return {
                'timeline': [],
                'narrative': 'No dated events found in the specified time range.',
                'key_dates': [],
                'date_range': {'start': start_year, 'end': end_year}
            }

        # Extract key events with descriptions
        key_events = []
        for item in temporal_data['timeline'][:50]:  # Limit for LLM context
            date = item['date']
            date_str = f"{date['year']}-{date['month']:02d}-{date['day']:02d}"

            event = {
                'date': date_str,
                'year': date['year'],
                'description': item['name'][:150],
                'type': item.get('type', 'event'),
                'fact_type': item.get('fact_type', '')
            }
            key_events.append(event)

        # Sort by date
        key_events.sort(key=lambda x: x['date'])

        # Build timeline text for LLM
        timeline_text = "\n".join([
            f"- {e['date']}: {e['description']}" +
            (f" [{e['fact_type']}]" if e.get('fact_type') else "")
            for e in key_events[:40]
        ])

        # Generate narrative summary using LLM
        prompt = f"""You are a legal analyst creating a chronological narrative of a legal case.

Based on the following timeline of events, create a clear, professional narrative summary suitable for a legal brief or case summary.

TIMELINE:
{timeline_text}

Instructions:
1. Organize events into logical phases (e.g., "Background", "Dispute", "Proceedings")
2. Use professional legal language
3. Highlight key milestones and turning points
4. Note any significant time gaps or delays
5. Keep the narrative concise but comprehensive (3-5 paragraphs)

Generate the narrative timeline:"""

        try:
            narrative_config = genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            )
            narrative = self._call_with_retry(prompt, narrative_config)
        except Exception as e:
            print(f"  Error generating narrative: {e}")
            narrative = "Timeline data available but narrative generation failed."

        # Identify key milestone dates
        key_dates = []
        seen_years = set()
        for event in key_events:
            if event['year'] not in seen_years:
                key_dates.append({
                    'date': event['date'],
                    'description': event['description'][:100]
                })
                seen_years.add(event['year'])

        return {
            'timeline': key_events,
            'narrative': narrative.strip(),
            'key_dates': key_dates[:10],
            'total_events': len(temporal_data['timeline']),
            'date_range': {
                'start': min(e['year'] for e in key_events) if key_events else None,
                'end': max(e['year'] for e in key_events) if key_events else None
            }
        }

    def suggest_related_questions(self, query: str, answer: str) -> List[str]:
        """
        Suggest follow-up questions based on the query and answer.

        Returns a list of related questions that a lawyer might ask next.
        """
        prompt = f"""Based on this legal knowledge graph query and answer, suggest 3-5 relevant follow-up questions a lawyer might ask.

Original Question: {query}

Answer Summary: {answer[:500]}

Suggest follow-up questions that:
1. Dig deeper into key entities mentioned
2. Explore relationships between parties
3. Investigate timelines or deadlines
4. Clarify facts or allegations
5. Compare different parties' positions

Output only the questions, one per line:"""

        try:
            config = genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=300,
            )
            response = self._call_with_retry(prompt, config)
            questions = [q.strip().lstrip('0123456789.-) ') for q in response.strip().split('\n') if q.strip()]
            return questions[:5]
        except Exception as e:
            print(f"  Error generating suggestions: {e}")
            return []

    # ========== INFERENCE METHODS ==========

    def get_important_entities(self, entity_types: List[str] = None, top_k: int = 20) -> List[Dict]:
        """
        Get the most important entities using PageRank-style importance scoring.

        Args:
            entity_types: Filter to specific types (e.g., ['Person', 'Organization'])
            top_k: Number of top entities to return

        Returns:
            List of entity importance info
        """
        importance = self.inference.compute_entity_importance(
            entity_types=entity_types,
            iterations=15
        )
        return [{
            'entity_id': e.entity_id,
            'name': e.entity_name,
            'type': e.entity_type,
            'importance_score': round(e.score, 4),
            'in_degree': e.components.get('in_degree', 0),
            'out_degree': e.components.get('out_degree', 0)
        } for e in importance[:top_k]]

    def get_fact_reliability(self, top_k: int = 30) -> List[Dict]:
        """
        Get fact reliability scores based on corroboration analysis.

        Returns:
            List of facts with reliability scores
        """
        corroboration = self.inference.assess_fact_corroboration(top_k=top_k)
        return [{
            'fact_id': f.fact_id,
            'text': f.fact_text,
            'type': f.fact_type,
            'reliability_score': round(f.corroboration_score, 3),
            'source_count': f.source_count,
            'has_contradictions': len(f.contradicting_facts) > 0
        } for f in corroboration]

    def get_inferred_relationships(self, entity_name: str) -> List[Dict]:
        """
        Get inferred (implicit) relationships for an entity.

        Args:
            entity_name: Name of the entity to analyze

        Returns:
            List of inferred relationships with confidence
        """
        matches = self.db.search_entities_by_name(entity_name, limit=1)
        if not matches:
            return []

        entity = matches[0]
        inferred = self.inference.infer_relationships(entity.id, max_hops=2)
        return inferred

    def resolve_entity_with_confidence(self, name: str, entity_type: str = None,
                                        context: List[str] = None) -> List[Dict]:
        """
        Resolve an entity name with Bayesian confidence scoring.

        Args:
            name: The name to resolve
            entity_type: Optional filter by type
            context: Optional context for disambiguation

        Returns:
            List of candidates with probability scores
        """
        candidates = self.inference.resolve_entity_bayesian(
            name, entity_type=entity_type, context=context
        )
        return [{
            'entity_id': c.entity_id,
            'name': c.canonical_name,
            'probability': round(c.probability, 4),
            'evidence': c.evidence
        } for c in candidates[:10]]
