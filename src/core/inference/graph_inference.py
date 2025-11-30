"""
Bayesian and probabilistic inference for the Knowledge Graph.

This module provides:
1. Entity importance scoring (PageRank-style)
2. Entity resolution with probabilistic confidence
3. Fact corroboration scoring (cross-document verification)
4. Contradiction detection
5. Relationship inference from graph patterns
"""
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

from ..storage.database import Database
from ..storage.models import Entity, Edge


@dataclass
class EntityImportance:
    """Entity importance score with breakdown."""
    entity_id: str
    entity_name: str
    entity_type: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class FactCorroboration:
    """Fact corroboration analysis."""
    fact_id: str
    fact_text: str
    fact_type: str
    corroboration_score: float
    source_count: int
    sources: List[str] = field(default_factory=list)
    contradicting_facts: List[Dict] = field(default_factory=list)


@dataclass
class EntityResolutionCandidate:
    """Candidate for entity resolution with probabilistic score."""
    entity_id: str
    canonical_name: str
    probability: float
    evidence: Dict[str, float] = field(default_factory=dict)


class GraphInference:
    """
    Probabilistic inference engine for the knowledge graph.

    Uses Bayesian reasoning and graph algorithms to:
    - Score entity importance
    - Resolve entity references
    - Assess fact reliability
    - Detect contradictions
    - Infer implicit relationships
    """

    def __init__(self, db: Database):
        self.db = db
        self._pagerank_cache = None
        self._entity_degree_cache = None

    # ========== ENTITY IMPORTANCE (PageRank-style) ==========

    def compute_entity_importance(self,
                                   damping: float = 0.85,
                                   iterations: int = 20,
                                   entity_types: List[str] = None) -> List[EntityImportance]:
        """
        Compute entity importance using a PageRank-inspired algorithm.

        Considers:
        - In-degree (how many entities point to this one)
        - Out-degree (how many entities this one points to)
        - Edge diversity (variety of relationship types)
        - Document mentions (grounding in source material)
        - Type-specific weights (Persons, Organizations more important for legal)

        Args:
            damping: PageRank damping factor (0.85 typical)
            iterations: Number of iterations for convergence
            entity_types: Filter to specific entity types

        Returns:
            List of EntityImportance sorted by score descending
        """
        # Get all entities
        entities = self.db.get_all_entities()
        if entity_types:
            entities = [e for e in entities if e.type in entity_types]

        if not entities:
            return []

        entity_ids = {e.id for e in entities}
        entity_map = {e.id: e for e in entities}
        n = len(entities)

        # Type-specific importance weights
        type_weights = {
            'Person': 1.5,
            'Organization': 1.5,
            'Fact': 1.2,
            'Date': 0.8,
            'Money': 1.0,
            'Location': 0.7,
            'Document': 0.5,
            'Reference': 0.3,
            'Clause': 0.6,
        }

        # Initialize scores
        scores = {e.id: 1.0 / n for e in entities}

        # Build adjacency list
        adj_in = defaultdict(list)  # who points to me
        adj_out = defaultdict(list)  # who I point to
        edge_types = defaultdict(set)  # variety of edge types per entity

        for entity in entities:
            edges_from = self.db.get_edges_from(entity.id)
            edges_to = self.db.get_edges_to(entity.id)

            for edge in edges_from:
                if edge.target_entity_id in entity_ids:
                    adj_out[entity.id].append(edge.target_entity_id)
                    edge_types[entity.id].add(edge.relation_type)

            for edge in edges_to:
                if edge.source_entity_id in entity_ids:
                    adj_in[entity.id].append(edge.source_entity_id)
                    edge_types[entity.id].add(edge.relation_type)

        # Get mention counts
        mention_counts = {}
        for entity in entities:
            mentions = self.db.get_mentions_for_entity(entity.id)
            mention_counts[entity.id] = len(mentions)

        # PageRank iterations
        for _ in range(iterations):
            new_scores = {}
            for entity_id in scores:
                # Base score from incoming links
                incoming_score = 0
                for source_id in adj_in[entity_id]:
                    out_degree = len(adj_out[source_id])
                    if out_degree > 0:
                        incoming_score += scores[source_id] / out_degree

                # PageRank formula with damping
                new_scores[entity_id] = (1 - damping) / n + damping * incoming_score

            scores = new_scores

        # Compute final importance with all factors
        results = []
        max_mentions = max(mention_counts.values()) if mention_counts.values() else 1

        for entity in entities:
            entity_id = entity.id

            # Component scores
            pagerank_score = scores[entity_id] * n  # Normalize to sum to n

            in_degree = len(adj_in[entity_id])
            out_degree = len(adj_out[entity_id])
            degree_score = math.log1p(in_degree + out_degree)

            edge_diversity = len(edge_types[entity_id])
            diversity_score = math.log1p(edge_diversity)

            mention_score = mention_counts[entity_id] / max_mentions

            type_weight = type_weights.get(entity.type, 1.0)

            # Combine scores with weights
            final_score = (
                0.35 * pagerank_score +
                0.20 * degree_score +
                0.15 * diversity_score +
                0.20 * mention_score +
                0.10 * type_weight
            )

            results.append(EntityImportance(
                entity_id=entity_id,
                entity_name=entity.canonical_name,
                entity_type=entity.type,
                score=final_score,
                components={
                    'pagerank': pagerank_score,
                    'degree': degree_score,
                    'diversity': diversity_score,
                    'mentions': mention_score,
                    'type_weight': type_weight,
                    'in_degree': in_degree,
                    'out_degree': out_degree
                }
            ))

        results.sort(key=lambda x: -x.score)
        return results

    # ========== ENTITY RESOLUTION (Bayesian) ==========

    def resolve_entity_bayesian(self,
                                 query_name: str,
                                 entity_type: str = None,
                                 context: List[str] = None) -> List[EntityResolutionCandidate]:
        """
        Resolve a query name to candidate entities using Bayesian inference.

        P(entity | query) ∝ P(query | entity) * P(entity)

        Where:
        - P(query | entity) = likelihood based on name similarity, aliases
        - P(entity) = prior based on entity importance/frequency

        Args:
            query_name: The name to resolve
            entity_type: Optional filter by entity type
            context: Optional context strings to help disambiguation

        Returns:
            List of candidates with probabilities
        """
        query_lower = query_name.lower().strip()

        # Get candidate entities
        candidates = self.db.search_entities_by_name(query_name, limit=50)
        if entity_type:
            candidates = [c for c in candidates if c.type == entity_type]

        if not candidates:
            return []

        results = []

        for entity in candidates:
            evidence = {}

            # P(query | entity) components

            # 1. Exact name match
            name_lower = entity.canonical_name.lower().strip()
            if query_lower == name_lower:
                evidence['exact_match'] = 1.0
            else:
                # Partial match score
                if query_lower in name_lower or name_lower in query_lower:
                    evidence['substring_match'] = 0.7
                else:
                    # Jaccard similarity of words
                    query_words = set(query_lower.split())
                    name_words = set(name_lower.split())
                    intersection = len(query_words & name_words)
                    union = len(query_words | name_words)
                    evidence['word_overlap'] = intersection / union if union > 0 else 0

            # 2. Alias match
            aliases = self.db.get_aliases(entity.id)
            for alias in aliases:
                alias_lower = alias.alias_text.lower().strip()
                if query_lower == alias_lower:
                    evidence['alias_exact'] = 0.9
                    break
                elif query_lower in alias_lower or alias_lower in query_lower:
                    evidence['alias_partial'] = 0.5

            # 3. Context match (if context provided)
            if context:
                context_text = ' '.join(context).lower()
                entity_text = (entity.canonical_name + ' ' +
                              str(entity.properties)).lower()
                context_words = set(context_text.split())
                entity_words = set(entity_text.split())
                context_match = len(context_words & entity_words)
                if context_match > 0:
                    evidence['context_match'] = min(context_match / 10, 1.0)

            # P(entity) - prior based on importance
            mentions = self.db.get_mentions_for_entity(entity.id)
            edges_from = self.db.get_edges_from(entity.id)
            edges_to = self.db.get_edges_to(entity.id)

            mention_prior = math.log1p(len(mentions)) / 10
            edge_prior = math.log1p(len(edges_from) + len(edges_to)) / 20
            evidence['mention_prior'] = mention_prior
            evidence['edge_prior'] = edge_prior

            # Confidence level prior
            confidence_prior = {
                'confirmed': 1.0,
                'extracted': 0.7,
                'inferred': 0.4
            }.get(entity.confidence, 0.5)
            evidence['confidence_prior'] = confidence_prior

            # Compute posterior probability (log-linear combination)
            weights = {
                'exact_match': 2.0,
                'substring_match': 1.2,
                'word_overlap': 1.0,
                'alias_exact': 1.5,
                'alias_partial': 0.8,
                'context_match': 0.6,
                'mention_prior': 0.4,
                'edge_prior': 0.3,
                'confidence_prior': 0.5
            }

            log_score = sum(weights.get(k, 0) * v for k, v in evidence.items())
            probability = 1 / (1 + math.exp(-log_score))  # Sigmoid

            results.append(EntityResolutionCandidate(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                probability=probability,
                evidence=evidence
            ))

        # Normalize probabilities
        total_prob = sum(r.probability for r in results)
        if total_prob > 0:
            for r in results:
                r.probability /= total_prob

        results.sort(key=lambda x: -x.probability)
        return results

    # ========== FACT CORROBORATION ==========

    def assess_fact_corroboration(self,
                                   fact_entity_id: str = None,
                                   top_k: int = 50) -> List[FactCorroboration]:
        """
        Assess how well facts are corroborated across documents.

        A fact is more reliable if:
        - Mentioned in multiple documents
        - Corroborated by related facts
        - Not contradicted by other facts

        Args:
            fact_entity_id: Specific fact to assess, or None for all
            top_k: Number of facts to analyze

        Returns:
            List of FactCorroboration results
        """
        # Get fact entities
        if fact_entity_id:
            fact = self.db.get_entity(fact_entity_id)
            facts = [fact] if fact and fact.type == 'Fact' else []
        else:
            facts = self.db.get_entities_by_type('Fact', limit=top_k)

        results = []

        for fact in facts:
            fact_text = fact.properties.get('full_text', fact.canonical_name)
            fact_type = fact.properties.get('fact_type', 'unknown')

            # Get document sources via mentions
            mentions = self.db.get_mentions_for_entity(fact.id)
            source_docs = set()
            for mention in mentions:
                doc = self.db.get_document(mention.doc_id)
                if doc:
                    source_docs.add(doc.filename)

            # Get provenance from edges
            edges_from = self.db.get_edges_from(fact.id)
            for edge in edges_from:
                if edge.provenance_doc_id:
                    doc = self.db.get_document(edge.provenance_doc_id)
                    if doc:
                        source_docs.add(doc.filename)

            # Check for contradicting facts
            contradictions = self._find_contradicting_facts(fact, facts)

            # Calculate corroboration score
            source_count = len(source_docs)
            contradiction_count = len(contradictions)

            # Bayesian-style scoring
            # P(fact is true | sources, contradictions)
            source_factor = 1 - math.exp(-0.5 * source_count)  # More sources = higher
            contradiction_factor = math.exp(-0.3 * contradiction_count)  # Contradictions = lower
            type_factor = {
                'finding': 0.9,  # Established findings
                'obligation': 0.95,  # Clear obligations
                'allegation': 0.6,  # Unproven claims
                'key_term': 0.85,
                'deadline': 0.9,
            }.get(fact_type, 0.7)

            corroboration_score = source_factor * contradiction_factor * type_factor

            results.append(FactCorroboration(
                fact_id=fact.id,
                fact_text=fact_text[:200],
                fact_type=fact_type,
                corroboration_score=corroboration_score,
                source_count=source_count,
                sources=list(source_docs)[:5],
                contradicting_facts=contradictions[:3]
            ))

        results.sort(key=lambda x: -x.corroboration_score)
        return results

    def _find_contradicting_facts(self,
                                   fact: Entity,
                                   all_facts: List[Entity]) -> List[Dict]:
        """
        Find facts that potentially contradict the given fact.

        Uses heuristics:
        - Same subject, different predicate values
        - Temporal inconsistencies
        - Conflicting amounts/dates
        """
        contradictions = []
        fact_text = fact.properties.get('full_text', fact.canonical_name).lower()

        # Extract key entities mentioned in fact
        fact_words = set(fact_text.split())

        for other in all_facts:
            if other.id == fact.id:
                continue

            other_text = other.properties.get('full_text', other.canonical_name).lower()
            other_words = set(other_text.split())

            # Check for word overlap (same subject)
            overlap = len(fact_words & other_words)
            if overlap < 5:  # Not enough overlap to be about same thing
                continue

            # Check for contradiction indicators
            contradiction_pairs = [
                ('did', 'did not'),
                ('was', 'was not'),
                ('is', 'is not'),
                ('has', 'has not'),
                ('failed', 'succeeded'),
                ('correct', 'incorrect'),
                ('true', 'false'),
                ('confirmed', 'denied'),
            ]

            for pos, neg in contradiction_pairs:
                if (pos in fact_text and neg in other_text) or \
                   (neg in fact_text and pos in other_text):
                    contradictions.append({
                        'fact_id': other.id,
                        'fact_text': other_text[:100],
                        'reason': f"Potential negation: '{pos}' vs '{neg}'"
                    })
                    break

            # Check for conflicting numbers (dates, amounts)
            fact_numbers = re.findall(r'\$[\d,]+|\d{4}|\d+%', fact_text)
            other_numbers = re.findall(r'\$[\d,]+|\d{4}|\d+%', other_text)

            if fact_numbers and other_numbers:
                # Same context but different numbers
                if overlap > 10 and set(fact_numbers) != set(other_numbers):
                    # Check if they're not already in contradictions
                    if not any(c['fact_id'] == other.id for c in contradictions):
                        contradictions.append({
                            'fact_id': other.id,
                            'fact_text': other_text[:100],
                            'reason': f"Conflicting values: {fact_numbers} vs {other_numbers}"
                        })

        return contradictions

    # ========== RELATIONSHIP INFERENCE ==========

    def infer_relationships(self,
                            entity_id: str,
                            max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Infer potential relationships not explicitly stated.

        Uses graph patterns:
        - Transitivity: A->B->C suggests A may relate to C
        - Common neighbors: A->X, B->X suggests A, B may relate
        - Type-based inference: Person affiliated_with Org, Org party_to Case

        Args:
            entity_id: Entity to find inferred relationships for
            max_hops: Maximum hops for transitive inference

        Returns:
            List of inferred relationships with confidence
        """
        entity = self.db.get_entity(entity_id)
        if not entity:
            return []

        inferred = []
        seen_pairs = set()

        # Get direct relationships
        direct_edges = self.db.get_edges_from(entity_id)
        direct_targets = {e.target_entity_id for e in direct_edges}

        # Transitive inference (A->B->C)
        for edge1 in direct_edges:
            second_hop_edges = self.db.get_edges_from(edge1.target_entity_id)
            for edge2 in second_hop_edges:
                if edge2.target_entity_id in direct_targets:
                    continue  # Already directly connected
                if edge2.target_entity_id == entity_id:
                    continue  # Self-loop

                pair = (entity_id, edge2.target_entity_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                target_entity = self.db.get_entity(edge2.target_entity_id)
                if not target_entity:
                    continue

                # Infer relationship type based on path
                inferred_type = self._infer_relation_type(
                    entity.type,
                    edge1.relation_type,
                    edge2.relation_type,
                    target_entity.type
                )

                # Calculate confidence based on edge strengths
                confidence = 0.5 * (
                    0.8 if edge1.confidence == 'confirmed' else 0.5
                ) * (
                    0.8 if edge2.confidence == 'confirmed' else 0.5
                )

                inferred.append({
                    'source_id': entity_id,
                    'source_name': entity.canonical_name,
                    'target_id': edge2.target_entity_id,
                    'target_name': target_entity.canonical_name,
                    'inferred_relation': inferred_type,
                    'confidence': confidence,
                    'path': [
                        {'entity': entity.canonical_name, 'relation': edge1.relation_type},
                        {'entity': self.db.get_entity(edge1.target_entity_id).canonical_name if self.db.get_entity(edge1.target_entity_id) else 'Unknown', 'relation': edge2.relation_type},
                        {'entity': target_entity.canonical_name}
                    ],
                    'inference_type': 'transitive'
                })

        # Common neighbor inference (A->X, B->X suggests A relates to B)
        for edge in direct_edges:
            # Find other entities pointing to same target
            incoming = self.db.get_edges_to(edge.target_entity_id)
            for other_edge in incoming:
                if other_edge.source_entity_id == entity_id:
                    continue
                if other_edge.source_entity_id in direct_targets:
                    continue  # Already directly connected

                pair = tuple(sorted([entity_id, other_edge.source_entity_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                other_entity = self.db.get_entity(other_edge.source_entity_id)
                if not other_entity:
                    continue

                # Calculate confidence based on shared neighbors
                shared_target = self.db.get_entity(edge.target_entity_id)
                confidence = 0.3  # Lower confidence for common neighbor

                if shared_target and shared_target.type in ['Organization', 'Document']:
                    confidence = 0.4  # Higher if shared context is important

                inferred.append({
                    'source_id': entity_id,
                    'source_name': entity.canonical_name,
                    'target_id': other_edge.source_entity_id,
                    'target_name': other_entity.canonical_name,
                    'inferred_relation': 'related_via_' + (shared_target.type.lower() if shared_target else 'unknown'),
                    'confidence': confidence,
                    'shared_context': shared_target.canonical_name if shared_target else None,
                    'inference_type': 'common_neighbor'
                })

        inferred.sort(key=lambda x: -x['confidence'])
        return inferred[:20]  # Limit results

    def _infer_relation_type(self,
                              source_type: str,
                              rel1: str,
                              rel2: str,
                              target_type: str) -> str:
        """Infer relationship type based on path pattern."""
        # Pattern-based inference
        patterns = {
            ('Person', 'employed_by', 'party_to', 'Organization'): 'indirectly_involved_with',
            ('Person', 'affiliated_with', 'party_to', 'Document'): 'may_be_mentioned_in',
            ('Organization', 'party_to', 'about', 'Fact'): 'subject_of_fact',
            ('Person', 'authored', 'about', 'Fact'): 'asserts',
        }

        pattern = (source_type, rel1, rel2, target_type)
        if pattern in patterns:
            return patterns[pattern]

        # Default inference
        return f'inferred_{rel1}_then_{rel2}'

    # ========== QUERY CONFIDENCE SCORING ==========

    def score_answer_confidence(self,
                                 entities: List[Entity],
                                 facts: List[Dict],
                                 edges: List[Edge]) -> Dict[str, Any]:
        """
        Score the confidence of a query answer based on supporting evidence.

        Args:
            entities: Entities supporting the answer
            facts: Facts supporting the answer
            edges: Relationships supporting the answer

        Returns:
            Confidence score and breakdown
        """
        if not entities and not facts:
            return {
                'confidence': 0.0,
                'reason': 'No supporting evidence found',
                'breakdown': {}
            }

        # Entity support score
        entity_confirmed = sum(1 for e in entities if e.confidence == 'confirmed')
        entity_extracted = sum(1 for e in entities if e.confidence == 'extracted')
        entity_score = (entity_confirmed * 1.0 + entity_extracted * 0.7) / max(len(entities), 1)

        # Fact support score
        fact_scores = []
        for fact in facts:
            fact_type = fact.get('type', 'unknown')
            type_score = {
                'finding': 0.9,
                'obligation': 0.95,
                'allegation': 0.5,  # Lower since unproven
                'key_term': 0.8,
            }.get(fact_type, 0.6)
            fact_scores.append(type_score)

        fact_score = sum(fact_scores) / max(len(fact_scores), 1)

        # Edge support score
        edge_confirmed = sum(1 for e in edges if e.confidence == 'confirmed')
        edge_extracted = sum(1 for e in edges if e.confidence == 'extracted')
        edge_score = (edge_confirmed * 1.0 + edge_extracted * 0.7) / max(len(edges), 1)

        # Coverage score (how much evidence)
        coverage = min(1.0, (len(entities) + len(facts)) / 10)

        # Overall confidence
        confidence = (
            0.30 * entity_score +
            0.35 * fact_score +
            0.15 * edge_score +
            0.20 * coverage
        )

        return {
            'confidence': round(confidence, 3),
            'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
            'breakdown': {
                'entity_support': round(entity_score, 3),
                'fact_support': round(fact_score, 3),
                'edge_support': round(edge_score, 3),
                'evidence_coverage': round(coverage, 3),
                'entity_count': len(entities),
                'fact_count': len(facts),
                'edge_count': len(edges)
            }
        }
