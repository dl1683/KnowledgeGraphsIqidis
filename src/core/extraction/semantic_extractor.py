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
    UNIFIED_EXTRACTION_PROMPT = """You are a legal document analyzer. Extract ALL entities, relationships, and facts from the following text in a SINGLE response.

OUTPUT FORMAT - Return a JSON object with three arrays:
{{
  "entities": [...],
  "relations": [...],
  "facts": [...]
}}

=== ENTITIES ===
For each entity, provide:
- name: Canonical name
- type: One of [Person, Organization, Location, Money, Date, Reference]
- properties: Relevant attributes (role, title, amount, currency, etc.)
- span_text: Exact text where entity appears

Entity Types:
- Person: Individuals mentioned by name
- Organization: Companies, firms, courts, agencies
- Location: Cities, states, countries, addresses
- Money: Dollar amounts, fees, damages
- Date: Specific dates
- Reference: Citations to cases, statutes, exhibits

=== RELATIONS ===
For each relationship, provide:
- source_name: Source entity name
- target_name: Target entity name
- relation_type: One of [represents, party_to, signed, employed_by, affiliated_with, testified, references, mentioned_in, about, binds, related_to]
- properties: Details about the relationship

=== FACTS ===
For each fact, provide:
- fact_type: One of [obligation, admission, deadline, key_term, allegation, finding, quote]
- text: The relevant text (verbatim or summarized)
- related_entities: List of entity names this fact relates to
- properties: Additional details

TEXT TO ANALYZE:
{text}

EXISTING ENTITIES (link to these if mentioned, avoid duplicates):
{existing_entities}

Output ONLY valid JSON with entities, relations, and facts arrays:
"""

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
        existing_str = "\n".join(f"- {e}" for e in existing_entities[:100]) if existing_entities else "None"

        # For large chunks, we can pass more text since Gemini has large context
        max_text_len = 50000  # ~20K tokens

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
        """Parse unified extraction response with robust JSON handling."""
        result = {'entities': [], 'relations': [], 'facts': []}

        if not response_text:
            return result

        text = response_text

        # Remove markdown code blocks if present
        if '```' in text:
            # Find content between ``` markers
            import re as _re
            code_match = _re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_match:
                text = code_match.group(1)
            else:
                # Fallback: strip ``` lines
                lines = text.split('\n')
                filtered = [l for l in lines if not l.strip().startswith('```')]
                text = '\n'.join(filtered)

        # Strip whitespace after removing code blocks
        text = text.strip()

        # Handle response that doesn't start with { (common Gemini behavior)
        if not text.startswith('{'):
            # Find the first JSON key
            first_key_pos = -1
            for key in ['"entities"', '"relations"', '"facts"']:
                pos = text.find(key)
                if pos != -1 and (first_key_pos == -1 or pos < first_key_pos):
                    first_key_pos = pos

            if first_key_pos != -1:
                # Wrap from the first key onwards with opening brace
                text = '{' + text[first_key_pos:]
            elif text.startswith('['):
                # It's an array - wrap as entities
                text = '{"entities": ' + text + '}'

        # Ensure closing brace if we have opening
        if text.startswith('{'):
            # Count braces to ensure proper closing
            open_braces = text.count('{')
            close_braces = text.count('}')
            if open_braces > close_braces:
                text = text.rstrip() + '}' * (open_braces - close_braces)

        # Try parsing the JSON
        try:
            parsed = json.loads(text)

            # Extract entities
            for e in parsed.get('entities', []):
                if isinstance(e, dict):
                    result['entities'].append(ExtractedEntity(
                        name=e.get('name', ''),
                        type=e.get('type', 'Unknown'),
                        properties=e.get('properties', {}),
                        span_text=e.get('span_text', e.get('name', '')),
                        confidence=e.get('confidence', 0.8)
                    ))

            # Extract relations
            for r in parsed.get('relations', []):
                if isinstance(r, dict):
                    result['relations'].append(ExtractedRelation(
                        source_name=r.get('source_name', ''),
                        target_name=r.get('target_name', ''),
                        relation_type=r.get('relation_type', 'related_to'),
                        properties=r.get('properties', {}),
                        confidence=r.get('confidence', 0.7)
                    ))

            # Extract facts
            for f in parsed.get('facts', []):
                if isinstance(f, dict):
                    result['facts'].append(ExtractedFact(
                        fact_type=f.get('fact_type', 'key_term'),
                        text=f.get('text', ''),
                        related_entities=f.get('related_entities', []),
                        properties=f.get('properties', {}),
                        confidence=f.get('confidence', 0.7)
                    ))

        except json.JSONDecodeError as e:
            print(f"JSON parse error in unified response: {e}")
            # Try to find JSON object in text
            obj_match = re.search(r'\{[\s\S]*\}', text)
            if obj_match:
                try:
                    parsed = json.loads(obj_match.group())
                    return self._parse_unified_response(json.dumps(parsed))
                except:
                    pass

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


class BatchSemanticExtractor:
    """Batch extraction for multiple chunks."""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.extractor = SemanticExtractor(api_key)

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
