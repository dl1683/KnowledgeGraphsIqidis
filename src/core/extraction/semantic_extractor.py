"""
Semantic extraction using Gemini Flash Lite.

Extracts:
- Entities (Person, Organization, Location, Money, etc.)
- Relations between entities
- Facts (obligations, admissions, deadlines, key terms)
"""
import json
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from json_repair import repair_json

from ..config import GEMINI_API_KEY, GEMINI_MODEL

# Rate limiting settings for free tier (15 requests/minute)
REQUESTS_PER_MINUTE = 15
MIN_DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # ~4 seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 45  # seconds


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    type: str  # Person, Organization, Location, Money, Date, Reference, Fact
    properties: Dict[str, Any]
    span_text: str
    confidence: float


@dataclass
class ExtractedRelation:
    """A relation between two entities."""
    source_name: str
    target_name: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float


@dataclass
class ExtractedFact:
    """A fact/assertion extracted from text."""
    fact_type: str  # obligation, admission, deadline, key_term, quote, allegation
    text: str
    related_entities: List[str]
    properties: Dict[str, Any]
    confidence: float


@dataclass
class SemanticExtraction:
    """Results from semantic extraction."""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    facts: List[ExtractedFact]


class SemanticExtractor:
    """Extract entities, relations, and facts using Gemini Flash Lite."""

    # Unified extraction prompt - extracts everything in one API call
    # Note: {{ and }} are escaped braces for Python's str.format()
    UNIFIED_EXTRACTION_PROMPT = """Extract ALL entities, relationships, and facts from this legal text. Be THOROUGH - capture every meaningful item.

OUTPUT FORMAT (JSON only, no markdown):
{{"entities":[...],"relations":[...],"facts":[...]}}

ENTITY EXTRACTION (no limit - extract ALL):
- name: canonical form (full name, not abbreviations)
- type: Person/Organization/Location/Money/Date/Document/Clause/Reference
- properties: {{role, title, amount, currency, context, aliases}}
- span_text: exact text where found
- confidence: 0.0-1.0

ENTITY TYPES TO EXTRACT:
- Person: individuals by name (include role: plaintiff, defendant, witness, attorney, judge, expert)
- Organization: companies, firms, courts, agencies, government bodies
- Location: cities, states, countries, addresses, jurisdictions
- Money: ALL dollar amounts, fees, damages, payments (include: amount, currency, purpose)
- Date: ALL dates mentioned (include: context - filing date, deadline, event date)
- Document: contracts, agreements, exhibits, motions, orders, filings
- Clause: specific contract sections, terms, provisions
- Reference: case citations, statute references, exhibit numbers

RELATIONSHIP EXTRACTION (capture ALL connections):
- source_name, target_name: exact entity names
- relation_type: represents/party_to/signed/employed_by/affiliated_with/testified/references/binds/related_to/owns/controls/parent_of/subsidiary_of/predecessor_of/successor_to/opposes/agrees_with/disputes/authored/filed/received/paid/owed
- properties: {{context, date, amount, document}}
- confidence: 0.0-1.0

INFER IMPLICIT RELATIONSHIPS:
- If A represents B in case X → A party_to X, B party_to X
- If A is CEO of B → A employed_by B, A controls B
- If A signed doc D as representative of B → A signed D, B party_to D
- If A paid B $X → A paid B (with amount), B received from A
- If A and B are opposing parties → A opposes B

FACT EXTRACTION (capture ALL significant facts):
- fact_type: obligation/admission/deadline/key_term/allegation/finding/ruling/quote/payment/breach/damage/claim
- text: the fact content (verbatim for quotes, summarized otherwise)
- related_entities: ALL entity names this fact involves
- properties: {{due_date, amount, source_document, paragraph, severity}}
- confidence: 0.0-1.0

FACT TYPES TO EXTRACT:
- obligation: duties, requirements ("shall", "must", "agrees to")
- deadline: time limits, due dates, statute of limitations
- allegation: claims, accusations made by parties
- finding: court findings, arbitrator determinations
- ruling: judge orders, decisions, judgments
- breach: alleged or proven contract breaches
- damage: harm claimed or proven
- payment: money transferred or owed
- key_term: important defined terms, thresholds, percentages
- quote: significant verbatim statements

TEXT:
{text}

EXISTING ENTITIES (link to these when mentioned, use their exact names):
{existing_entities}

Output compact JSON only:"""

    ENTITY_EXTRACTION_PROMPT = """You are a legal document analyzer. Extract all entities from the following text chunk.

For each entity, identify:
- name: The canonical name of the entity
- type: One of [Person, Organization, Location, Money, Date, Reference]
- properties: Any relevant attributes (e.g., role, title, amount, currency)
- span_text: The exact text where this entity appears

IMPORTANT GUIDELINES:
1. Person: Real individuals mentioned by name (not pronouns)
2. Organization: Companies, firms, courts, agencies, etc.
3. Location: Cities, states, countries, addresses
4. Money: Dollar amounts, fees, damages (include currency and amount)
5. Date: Specific dates mentioned
6. Reference: Citations to cases, statutes, exhibits, documents

Output JSON array of entities. Be thorough but avoid duplicates.

TEXT:
{text}

EXISTING ENTITIES IN GRAPH (avoid duplicates, link to these if mentioned):
{existing_entities}

Output only valid JSON array:
"""

    RELATION_EXTRACTION_PROMPT = """You are a legal document analyzer. Extract relationships between entities in the following text.

For each relationship, identify:
- source_name: The source entity name
- target_name: The target entity name
- relation_type: One of [represents, party_to, signed, employed_by, affiliated_with, testified, references, mentioned_in, about, binds, related_to]
- properties: Any relevant details about the relationship

IMPORTANT GUIDELINES:
1. represents: Attorney/firm represents client
2. party_to: Entity is party to agreement/case
3. signed: Entity signed document
4. employed_by: Person works for organization
5. affiliated_with: Entity connected to another
6. testified: Person testified about something
7. references: Document/clause references another
8. binds: Obligation/agreement binds entity
9. related_to: General relationship (specify in properties)

TEXT:
{text}

ENTITIES FOUND:
{entities}

Output only valid JSON array of relationships:
"""

    FACT_EXTRACTION_PROMPT = """You are a legal document analyzer. Extract important facts, obligations, and assertions from the following text.

For each fact, identify:
- fact_type: One of [obligation, admission, deadline, key_term, allegation, finding, quote]
- text: The relevant text (verbatim or summarized)
- related_entities: List of entity names this fact relates to
- properties: Additional details (e.g., due_date for deadlines, monetary_value for damages)

WHAT TO EXTRACT:
1. OBLIGATION: Duties, requirements ("shall", "must", "required to", "agrees to")
   - Include: who is obligated, to do what, by when, consequences
2. ADMISSION: Acknowledgments, concessions
   - Include: who admitted, what was admitted
3. DEADLINE: Time limits, due dates
   - Include: what is due, when, consequences of missing
4. KEY_TERM: Important defined terms, amounts, percentages
   - Include: the term and its significance
5. ALLEGATION: Claims, accusations
   - Include: who alleges, against whom, the claim
6. FINDING: Court findings, determinations
   - Include: who found, what was found
7. QUOTE: Important verbatim statements worth preserving

TEXT:
{text}

ENTITIES IN THIS CHUNK:
{entities}

Output only valid JSON array of facts:
"""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.last_request_time = 0

        # Generation config for structured output
        self.generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent extraction
            top_p=0.95,
            max_output_tokens=16384,  # Increased for unified extraction
        )

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
            sleep_time = MIN_DELAY_BETWEEN_REQUESTS - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _call_with_retry(self, prompt: str) -> str:
        """Call the API with retry logic for rate limits."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower():
                    # Rate limit hit - wait and retry
                    wait_time = RETRY_BASE_DELAY * (attempt + 1)
                    print(f"    Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(f"Failed after {MAX_RETRIES} retries")

    def extract(self, text: str, existing_entities: List[str] = None) -> SemanticExtraction:
        """Extract entities, relations, and facts from text using unified prompt (1 API call)."""
        existing_entities = existing_entities or []

        # Use unified extraction - 1 API call instead of 3
        return self._extract_unified(text, existing_entities)

    def _extract_unified(self, text: str, existing_entities: List[str]) -> SemanticExtraction:
        """Extract everything in a single API call for efficiency."""
        existing_str = ", ".join(existing_entities[:50]) if existing_entities else "None"

        # Reduced text length to prevent response truncation
        max_text_len = 25000  # ~10K tokens - smaller to fit response in output limit

        prompt = self.UNIFIED_EXTRACTION_PROMPT.format(
            text=text[:max_text_len],
            existing_entities=existing_str
        )

        try:
            response_text = self._call_with_retry(prompt)

            # Debug: Check response format
            if response_text:
                first_chars = repr(response_text[:50]) if len(response_text) > 50 else repr(response_text)
                # print(f"    [DEBUG] Response starts with: {first_chars}")

            # Parse unified JSON response
            result = self._parse_unified_response(response_text)

            return SemanticExtraction(
                entities=result['entities'],
                relations=result['relations'],
                facts=result['facts']
            )

        except Exception as ex:
            import traceback
            print(f"Unified extraction error: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            # Fallback to empty results
            return SemanticExtraction(entities=[], relations=[], facts=[])

    def _parse_unified_response(self, response_text: str) -> Dict[str, List]:
        """Parse unified extraction response with robust JSON handling using json_repair."""
        result = {'entities': [], 'relations': [], 'facts': []}

        if not response_text:
            return result

        text = response_text

        # Remove markdown code blocks if present
        if '```' in text:
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_match:
                text = code_match.group(1)
            else:
                lines = text.split('\n')
                filtered = [l for l in lines if not l.strip().startswith('```')]
                text = '\n'.join(filtered)

        text = text.strip()

        # Handle response that doesn't start with {
        if not text.startswith('{'):
            first_key_pos = -1
            for key in ['"entities"', '"relations"', '"facts"']:
                pos = text.find(key)
                if pos != -1 and (first_key_pos == -1 or pos < first_key_pos):
                    first_key_pos = pos

            if first_key_pos != -1:
                text = '{' + text[first_key_pos:]
            elif text.startswith('['):
                text = '{"entities": ' + text + '}'

        # Use json_repair to fix truncated/malformed JSON
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                parsed = repaired
            else:
                # If repair returns a string, try parsing it
                parsed = json.loads(repaired) if isinstance(repaired, str) else {}
        except Exception as repair_error:
            # Fallback: try standard JSON parsing
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"JSON parse error (repair failed): {e}")
                return result

        # Extract entities
        for e in parsed.get('entities', []):
            if isinstance(e, dict) and e.get('name'):
                result['entities'].append(ExtractedEntity(
                    name=e.get('name', ''),
                    type=e.get('type', 'Unknown'),
                    properties=e.get('properties', {}),
                    span_text=e.get('span_text', e.get('name', '')),
                    confidence=e.get('confidence', 0.8)
                ))

        # Extract relations
        for r in parsed.get('relations', []):
            if isinstance(r, dict) and r.get('source_name') and r.get('target_name'):
                result['relations'].append(ExtractedRelation(
                    source_name=r.get('source_name', ''),
                    target_name=r.get('target_name', ''),
                    relation_type=r.get('relation_type', 'related_to'),
                    properties=r.get('properties', {}),
                    confidence=r.get('confidence', 0.7)
                ))

        # Extract facts
        for f in parsed.get('facts', []):
            if isinstance(f, dict) and f.get('text'):
                result['facts'].append(ExtractedFact(
                    fact_type=f.get('fact_type', 'key_term'),
                    text=f.get('text', ''),
                    related_entities=f.get('related_entities', []),
                    properties=f.get('properties', {}),
                    confidence=f.get('confidence', 0.7)
                ))

        return result

    def extract_legacy(self, text: str, existing_entities: List[str] = None) -> SemanticExtraction:
        """Legacy extraction using 3 separate API calls (kept for fallback)."""
        existing_entities = existing_entities or []

        # Step 1: Extract entities
        entities = self._extract_entities(text, existing_entities)

        # Step 2: Extract relations (using found entities)
        entity_names = [e.name for e in entities]
        relations = self._extract_relations(text, entity_names)

        # Step 3: Extract facts
        facts = self._extract_facts(text, entity_names)

        return SemanticExtraction(
            entities=entities,
            relations=relations,
            facts=facts
        )

    def _extract_entities(self, text: str, existing_entities: List[str]) -> List[ExtractedEntity]:
        """Extract entities from text."""
        existing_str = "\n".join(f"- {e}" for e in existing_entities[:50]) if existing_entities else "None"

        prompt = self.ENTITY_EXTRACTION_PROMPT.format(
            text=text[:6000],  # Limit text length
            existing_entities=existing_str
        )

        try:
            response_text = self._call_with_retry(prompt)

            # Parse JSON response
            entities = self._parse_json_response(response_text)

            result = []
            for e in entities:
                if not isinstance(e, dict):
                    continue

                result.append(ExtractedEntity(
                    name=e.get('name', ''),
                    type=e.get('type', 'Unknown'),
                    properties=e.get('properties', {}),
                    span_text=e.get('span_text', e.get('name', '')),
                    confidence=e.get('confidence', 0.8)
                ))

            return result

        except Exception as ex:
            print(f"Entity extraction error: {ex}")
            return []

    def _extract_relations(self, text: str, entity_names: List[str]) -> List[ExtractedRelation]:
        """Extract relations between entities."""
        if not entity_names:
            return []

        entities_str = "\n".join(f"- {name}" for name in entity_names)

        prompt = self.RELATION_EXTRACTION_PROMPT.format(
            text=text[:6000],
            entities=entities_str
        )

        try:
            response_text = self._call_with_retry(prompt)

            relations = self._parse_json_response(response_text)

            result = []
            for r in relations:
                if not isinstance(r, dict):
                    continue

                result.append(ExtractedRelation(
                    source_name=r.get('source_name', ''),
                    target_name=r.get('target_name', ''),
                    relation_type=r.get('relation_type', 'related_to'),
                    properties=r.get('properties', {}),
                    confidence=r.get('confidence', 0.7)
                ))

            return result

        except Exception as ex:
            print(f"Relation extraction error: {ex}")
            return []

    def _extract_facts(self, text: str, entity_names: List[str]) -> List[ExtractedFact]:
        """Extract facts and assertions from text."""
        entities_str = "\n".join(f"- {name}" for name in entity_names) if entity_names else "None identified"

        prompt = self.FACT_EXTRACTION_PROMPT.format(
            text=text[:6000],
            entities=entities_str
        )

        try:
            response_text = self._call_with_retry(prompt)

            facts = self._parse_json_response(response_text)

            result = []
            for f in facts:
                if not isinstance(f, dict):
                    continue

                result.append(ExtractedFact(
                    fact_type=f.get('fact_type', 'key_term'),
                    text=f.get('text', ''),
                    related_entities=f.get('related_entities', []),
                    properties=f.get('properties', {}),
                    confidence=f.get('confidence', 0.7)
                ))

            return result

        except Exception as ex:
            print(f"Fact extraction error: {ex}")
            return []

    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """Parse JSON from LLM response, handling various formats."""
        if not response_text:
            return []

        # Try to find JSON array in response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith('```'):
            # Find the content between ``` markers
            lines = text.split('\n')
            start_idx = 1 if lines[0].startswith('```') else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '```':
                    end_idx = i
                    break
            text = '\n'.join(lines[start_idx:end_idx])

        # Try to parse as JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # Sometimes LLM wraps in object
                for key in ['entities', 'relations', 'facts', 'results', 'data']:
                    if key in result and isinstance(result[key], list):
                        return result[key]
                return [result]
        except json.JSONDecodeError:
            pass

        # Try to find array in text
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        # Try line-by-line JSON objects
        results = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return results


class RelationshipInferrer:
    """Infer implicit relationships from extracted entities and facts."""

    # Patterns for inferring relationships
    ROLE_TO_RELATION = {
        'plaintiff': ('party_to', 'opposes'),
        'defendant': ('party_to', 'opposes'),
        'claimant': ('party_to', 'opposes'),
        'respondent': ('party_to', 'opposes'),
        'petitioner': ('party_to', None),
        'attorney': ('represents', None),
        'counsel': ('represents', None),
        'lawyer': ('represents', None),
        'judge': ('presides_over', None),
        'arbitrator': ('presides_over', None),
        'witness': ('testified_in', None),
        'expert': ('testified_in', None),
        'ceo': ('controls', 'employed_by'),
        'president': ('controls', 'employed_by'),
        'director': ('controls', 'employed_by'),
        'officer': ('employed_by', None),
        'employee': ('employed_by', None),
        'shareholder': ('owns', None),
        'owner': ('owns', 'controls'),
        'subsidiary': ('subsidiary_of', None),
        'parent': ('parent_of', None),
        'successor': ('successor_to', None),
        'predecessor': ('predecessor_of', None),
    }

    @staticmethod
    def infer_relationships(
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        facts: List[ExtractedFact]
    ) -> List[ExtractedRelation]:
        """Infer additional relationships from extracted data."""
        inferred = []
        existing_pairs = {(r.source_name.lower(), r.target_name.lower(), r.relation_type) for r in relations}

        # Build entity lookup
        entity_by_name = {e.name.lower(): e for e in entities}
        orgs = [e for e in entities if e.type == 'Organization']
        people = [e for e in entities if e.type == 'Person']
        documents = [e for e in entities if e.type in ('Document', 'Reference')]

        # 1. Infer from entity roles
        for entity in entities:
            props = entity.properties if isinstance(entity.properties, dict) else {}
            role = props.get('role', '').lower()

            if role in RelationshipInferrer.ROLE_TO_RELATION:
                rel_type, secondary_rel = RelationshipInferrer.ROLE_TO_RELATION[role]

                # For party roles, link to organizations or the case
                if role in ('plaintiff', 'defendant', 'claimant', 'respondent'):
                    for doc in documents:
                        if 'case' in doc.name.lower() or 'v.' in doc.name or 'vs' in doc.name.lower():
                            pair = (entity.name.lower(), doc.name.lower(), 'party_to')
                            if pair not in existing_pairs:
                                inferred.append(ExtractedRelation(
                                    source_name=entity.name,
                                    target_name=doc.name,
                                    relation_type='party_to',
                                    properties={'inferred': True, 'role': role},
                                    confidence=0.7
                                ))
                                existing_pairs.add(pair)

                # For attorney roles, link to clients (other parties)
                elif role in ('attorney', 'counsel', 'lawyer'):
                    # Try to infer client from context
                    client_hint = props.get('client', props.get('for', props.get('representing', '')))
                    if client_hint:
                        pair = (entity.name.lower(), client_hint.lower(), 'represents')
                        if pair not in existing_pairs:
                            inferred.append(ExtractedRelation(
                                source_name=entity.name,
                                target_name=client_hint,
                                relation_type='represents',
                                properties={'inferred': True},
                                confidence=0.6
                            ))
                            existing_pairs.add(pair)

                # For executive roles, link to organization
                elif role in ('ceo', 'president', 'director', 'officer'):
                    org_hint = props.get('company', props.get('organization', props.get('of', '')))
                    if org_hint:
                        pair = (entity.name.lower(), org_hint.lower(), 'employed_by')
                        if pair not in existing_pairs:
                            inferred.append(ExtractedRelation(
                                source_name=entity.name,
                                target_name=org_hint,
                                relation_type='employed_by',
                                properties={'inferred': True, 'role': role},
                                confidence=0.8
                            ))
                            existing_pairs.add(pair)

        # 2. Infer opposing party relationships
        plaintiffs = [e for e in entities if e.properties.get('role', '').lower() in ('plaintiff', 'claimant')]
        defendants = [e for e in entities if e.properties.get('role', '').lower() in ('defendant', 'respondent')]

        for p in plaintiffs:
            for d in defendants:
                pair = (p.name.lower(), d.name.lower(), 'opposes')
                if pair not in existing_pairs:
                    inferred.append(ExtractedRelation(
                        source_name=p.name,
                        target_name=d.name,
                        relation_type='opposes',
                        properties={'inferred': True, 'context': 'opposing parties'},
                        confidence=0.9
                    ))
                    existing_pairs.add(pair)

        # 3. Infer from facts
        for fact in facts:
            related = fact.related_entities if isinstance(fact.related_entities, list) else []
            props = fact.properties if isinstance(fact.properties, dict) else {}

            # Payment facts -> paid/received relationships
            if fact.fact_type in ('payment', 'paid'):
                if len(related) >= 2:
                    pair = (related[0].lower() if isinstance(related[0], str) else '',
                           related[1].lower() if isinstance(related[1], str) else '', 'paid')
                    if pair[0] and pair[1] and pair not in existing_pairs:
                        inferred.append(ExtractedRelation(
                            source_name=related[0] if isinstance(related[0], str) else str(related[0]),
                            target_name=related[1] if isinstance(related[1], str) else str(related[1]),
                            relation_type='paid',
                            properties={'inferred': True, 'amount': props.get('amount', '')},
                            confidence=0.7
                        ))
                        existing_pairs.add(pair)

            # Breach facts -> breached relationship
            elif fact.fact_type == 'breach':
                for entity_ref in related:
                    entity_name = entity_ref if isinstance(entity_ref, str) else str(entity_ref)
                    # Find the contract/agreement
                    for doc in documents:
                        if any(word in doc.name.lower() for word in ['agreement', 'contract', 'covenant']):
                            pair = (entity_name.lower(), doc.name.lower(), 'breached')
                            if pair not in existing_pairs:
                                inferred.append(ExtractedRelation(
                                    source_name=entity_name,
                                    target_name=doc.name,
                                    relation_type='breached',
                                    properties={'inferred': True, 'fact': fact.text[:100]},
                                    confidence=0.6
                                ))
                                existing_pairs.add(pair)

            # Obligation facts -> binds relationship
            elif fact.fact_type == 'obligation':
                for entity_ref in related:
                    entity_name = entity_ref if isinstance(entity_ref, str) else str(entity_ref)
                    for doc in documents:
                        pair = (doc.name.lower(), entity_name.lower(), 'binds')
                        if pair not in existing_pairs:
                            inferred.append(ExtractedRelation(
                                source_name=doc.name,
                                target_name=entity_name,
                                relation_type='binds',
                                properties={'inferred': True, 'obligation': fact.text[:100]},
                                confidence=0.6
                            ))
                            existing_pairs.add(pair)

        # 4. Infer corporate relationships from naming patterns
        for org in orgs:
            org_name_lower = org.name.lower()
            for other_org in orgs:
                if org.name == other_org.name:
                    continue
                other_name_lower = other_org.name.lower()

                # Check for parent/subsidiary patterns
                if org_name_lower in other_name_lower or other_name_lower in org_name_lower:
                    # Longer name is likely the full/parent entity
                    if len(org.name) > len(other_org.name):
                        pair = (other_org.name.lower(), org.name.lower(), 'affiliated_with')
                    else:
                        pair = (org.name.lower(), other_org.name.lower(), 'affiliated_with')

                    if pair not in existing_pairs:
                        inferred.append(ExtractedRelation(
                            source_name=pair[0],
                            target_name=pair[1],
                            relation_type='affiliated_with',
                            properties={'inferred': True, 'reason': 'name_similarity'},
                            confidence=0.5
                        ))
                        existing_pairs.add(pair)

        return inferred


class BatchSemanticExtractor:
    """Batch extraction for multiple chunks."""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.extractor = SemanticExtractor(api_key)
        self.inferrer = RelationshipInferrer()

    def extract_from_chunks(self, chunks: List[str], existing_entities: List[str] = None) -> SemanticExtraction:
        """Extract from multiple chunks and merge results."""
        all_entities = []
        all_relations = []
        all_facts = []

        existing = existing_entities or []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")

            extraction = self.extractor.extract(chunk, existing)

            all_entities.extend(extraction.entities)
            all_relations.extend(extraction.relations)
            all_facts.extend(extraction.facts)

            # Add newly found entities to existing list for next chunk
            new_names = [e.name for e in extraction.entities]
            existing.extend(new_names)

        # Deduplicate entities by name
        seen_entities = {}
        for e in all_entities:
            name_lower = e.name.lower()
            if name_lower not in seen_entities:
                seen_entities[name_lower] = e
            elif e.confidence > seen_entities[name_lower].confidence:
                seen_entities[name_lower] = e

        return SemanticExtraction(
            entities=list(seen_entities.values()),
            relations=all_relations,
            facts=all_facts
        )
