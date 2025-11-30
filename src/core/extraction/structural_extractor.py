"""
Structural extraction for legal documents.

Extracts high-confidence structural elements:
- Defined terms (explicit aliases)
- Parties from caption blocks
- Signature blocks
- Document metadata
"""
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DefinedTerm:
    """A defined term found in a document."""
    term: str
    definition: str
    aliases: List[str]
    span_start: int
    span_end: int


@dataclass
class Party:
    """A party identified in a document."""
    name: str
    role: str  # plaintiff, defendant, claimant, respondent, buyer, seller, etc.
    aliases: List[str]
    span_start: int
    span_end: int


@dataclass
class StructuralExtraction:
    """Results from structural extraction."""
    defined_terms: List[DefinedTerm]
    parties: List[Party]
    document_type: str
    key_dates: List[Dict[str, Any]]
    case_number: str
    court_or_tribunal: str


class StructuralExtractor:
    """Extract structural elements from legal documents using pattern matching."""

    # Common party role patterns - restrictive to avoid matching sentence fragments
    # Pattern: Organization name (max 60 chars, letters, spaces, &, periods for abbreviations)
    _ORG_NAME = r'([A-Z][A-Za-z&\s\.]{2,58}(?:Inc\.|Corp\.|LLC|LLP|Ltd\.|Corporation|Company)?)'

    PARTY_PATTERNS = [
        # Litigation patterns - match "Name, Claimant" or "Name as Claimant"
        (rf'{_ORG_NAME}\s*,\s*Plaintiff\b', 'plaintiff'),
        (rf'{_ORG_NAME}\s*,\s*Defendant\b', 'defendant'),
        (rf'{_ORG_NAME}\s*,\s*Claimant\b', 'claimant'),
        (rf'{_ORG_NAME}\s*,\s*Respondent\b', 'respondent'),
        (rf'{_ORG_NAME}\s*,\s*Petitioner\b', 'petitioner'),
        (rf'{_ORG_NAME}\s*,\s*Appellant\b', 'appellant'),
        (rf'{_ORG_NAME}\s*,\s*Appellee\b', 'appellee'),

        # Contract patterns with defined term - match 'Name (the "Buyer")'
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Buyer["\']\s*\)', 'buyer'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Seller["\']\s*\)', 'seller'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Lessor["\']\s*\)', 'lessor'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Lessee["\']\s*\)', 'lessee'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Licensor["\']\s*\)', 'licensor'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Licensee["\']\s*\)', 'licensee'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Borrower["\']\s*\)', 'borrower'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Lender["\']\s*\)', 'lender'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Company["\']\s*\)', 'company'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Customer["\']\s*\)', 'customer'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Vendor["\']\s*\)', 'vendor'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Contractor["\']\s*\)', 'contractor'),
        (rf'{_ORG_NAME}\s*\(\s*(?:the\s+)?["\']Client["\']\s*\)', 'client'),
    ]

    # Defined term patterns
    DEFINED_TERM_PATTERNS = [
        # "Term" means X
        r'["\']([A-Z][A-Za-z\s]+)["\']?\s+(?:means?|shall mean|refers? to|is defined as)\s+([^.;]+[.;])',
        # (the "Term")
        r'\((?:the\s+)?["\']([A-Z][A-Za-z\s]+)["\']\)',
        # hereinafter "Term" or hereinafter referred to as "Term"
        r'hereinafter\s+(?:referred to as\s+)?["\']([A-Z][A-Za-z\s]+)["\']',
        # collectively, the "Terms"
        r'collectively,?\s+(?:the\s+)?["\']([A-Z][A-Za-z\s]+)["\']',
    ]

    # Date patterns
    DATE_PATTERNS = [
        # January 1, 2024
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
        # 01/01/2024 or 1/1/2024
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        # 2024-01-01
        r'(\d{4}-\d{2}-\d{2})',
        # 1st day of January, 2024
        r'(\d{1,2}(?:st|nd|rd|th)\s+day\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})',
    ]

    # Case number patterns
    CASE_NUMBER_PATTERNS = [
        r'(?:Case|Cause|Docket|Matter)\s*(?:No\.?|Number|#)\s*[:.]?\s*([A-Z0-9\-:]+)',
        r'(\d{1,2}[:-]cv[:-]\d+)',  # Federal civil case
        r'(\d{1,2}[:-]cr[:-]\d+)',  # Federal criminal case
        r'([A-Z]{2,3}\s*\d{4}[:-]\d+)',  # State case patterns
    ]

    def __init__(self):
        pass

    def extract(self, text: str) -> StructuralExtraction:
        """Extract structural elements from document text."""
        defined_terms = self._extract_defined_terms(text)
        parties = self._extract_parties(text)
        document_type = self._detect_document_type(text)
        key_dates = self._extract_dates(text)
        case_number = self._extract_case_number(text)
        court = self._extract_court(text)

        return StructuralExtraction(
            defined_terms=defined_terms,
            parties=parties,
            document_type=document_type,
            key_dates=key_dates,
            case_number=case_number,
            court_or_tribunal=court
        )

    def _extract_defined_terms(self, text: str) -> List[DefinedTerm]:
        """Extract defined terms from text."""
        defined_terms = []
        seen_terms = set()

        for pattern in self.DEFINED_TERM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(1).strip()

                # Skip duplicates and common false positives
                term_lower = term.lower()
                if term_lower in seen_terms:
                    continue
                if len(term) < 2 or len(term) > 50:
                    continue
                if term_lower in ['the', 'this', 'that', 'such', 'said', 'any', 'all']:
                    continue

                seen_terms.add(term_lower)

                # Get definition if available
                definition = match.group(2).strip() if len(match.groups()) > 1 else ""

                defined_terms.append(DefinedTerm(
                    term=term,
                    definition=definition[:500] if definition else "",
                    aliases=[term],
                    span_start=match.start(),
                    span_end=match.end()
                ))

        return defined_terms

    def _extract_parties(self, text: str) -> List[Party]:
        """Extract parties from document."""
        parties = []
        seen_names = set()

        # First, look for explicit party sections
        party_section = self._find_party_section(text)
        if party_section:
            text_to_search = party_section
        else:
            # Search in first 3000 chars (usually contains caption/intro)
            text_to_search = text[:3000]

        for pattern, role in self.PARTY_PATTERNS:
            for match in re.finditer(pattern, text_to_search, re.IGNORECASE):
                name = match.group(1).strip()

                # Clean up name
                name = re.sub(r'\s+', ' ', name)
                name = name.strip(' ,.')

                # Skip if too short, too long, or already seen
                if len(name) < 3 or len(name) > 80 or name.lower() in seen_names:
                    continue

                # Skip if too many words (likely a sentence fragment)
                word_count = len(name.split())
                if word_count > 8:
                    continue

                # Skip generic terms and common false positives
                skip_terms = ['the', 'this', 'that', 'party', 'parties', 'pursuant',
                              'statement', 'claim', 'amended', 'demand', 'arbitration',
                              'resolution', 'rules', 'procedures', 'against']
                if name.lower() in skip_terms:
                    continue

                # Skip if it looks like a sentence (has common verbs)
                sentence_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'hereby', 'submits', 'brings']
                if any(word.lower() in sentence_indicators for word in name.split()):
                    continue

                seen_names.add(name.lower())

                # Find aliases (the defined term alias like "Buyer")
                aliases = [name]
                alias_match = re.search(
                    rf'{re.escape(name)}.*?["\']([A-Za-z]+)["\']',
                    text_to_search,
                    re.IGNORECASE
                )
                if alias_match:
                    aliases.append(alias_match.group(1))
                else:
                    # Add role as potential alias (e.g., "Buyer", "Defendant")
                    aliases.append(role.capitalize())

                parties.append(Party(
                    name=name,
                    role=role,
                    aliases=list(set(aliases)),
                    span_start=match.start(),
                    span_end=match.end()
                ))

        return parties

    def _find_party_section(self, text: str) -> str:
        """Find the party/caption section of a document."""
        # Look for common section headers
        section_patterns = [
            r'PARTIES\s*\n([\s\S]{0,2000}?)(?=\n[A-Z]{3,}|\n\d+\.\s)',
            r'THE PARTIES\s*\n([\s\S]{0,2000}?)(?=\n[A-Z]{3,}|\n\d+\.\s)',
            r'(?:BETWEEN|By and Between)[:\s]*([\s\S]{0,1500}?)(?=\n[A-Z]{3,}|\nWHEREAS)',
        ]

        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document."""
        text_lower = text[:2000].lower()

        type_indicators = {
            'complaint': ['complaint', 'plaintiff hereby alleges', 'plaintiff brings this action'],
            'answer': ['answer to complaint', 'defendant answers', 'defendant hereby answers'],
            'motion': ['motion to', 'moves this court', 'motion for'],
            'brief': ['brief in support', 'memorandum of law', 'legal memorandum', 'pre-hearing brief', 'prehearing brief'],
            'contract': ['agreement', 'contract', 'hereby agree', 'terms and conditions'],
            'deposition': ['deposition of', 'deposition transcript', 'q.', 'a.'],
            'affidavit': ['affidavit', 'being duly sworn', 'swear under penalty'],
            'witness_statement': ['witness statement', 'statement of'],
            'expert_report': ['expert report', 'expert opinion', 'expert witness'],
            'discovery': ['request for production', 'interrogatories', 'request for admission'],
            'order': ['order of the court', 'it is hereby ordered', 'so ordered'],
            'statement_of_claim': ['statement of claim', 'claimant states'],
            'settlement': ['settlement agreement', 'settlement', 'compromise'],
        }

        for doc_type, indicators in type_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return doc_type

        return 'unknown'

    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract key dates from document."""
        dates = []
        seen = set()

        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                date_str = match.group(1)
                if date_str not in seen:
                    seen.add(date_str)

                    # Try to identify date context
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]

                    date_type = self._classify_date(context)

                    dates.append({
                        'date': date_str,
                        'type': date_type,
                        'span_start': match.start(),
                        'span_end': match.end(),
                        'context': context
                    })

        return dates[:20]  # Limit to first 20 dates

    def _classify_date(self, context: str) -> str:
        """Classify the type of date based on surrounding context."""
        context_lower = context.lower()

        if any(w in context_lower for w in ['effective', 'commence', 'begin', 'start']):
            return 'effective_date'
        elif any(w in context_lower for w in ['expire', 'termination', 'end']):
            return 'expiration_date'
        elif any(w in context_lower for w in ['sign', 'execute', 'dated']):
            return 'execution_date'
        elif any(w in context_lower for w in ['due', 'deadline', 'by']):
            return 'deadline'
        elif any(w in context_lower for w in ['file', 'filed']):
            return 'filing_date'
        else:
            return 'date'

    def _extract_case_number(self, text: str) -> str:
        """Extract case/matter number."""
        # Search in first 1000 chars
        text_to_search = text[:1000]

        for pattern in self.CASE_NUMBER_PATTERNS:
            match = re.search(pattern, text_to_search, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _extract_court(self, text: str) -> str:
        """Extract court or tribunal name."""
        text_to_search = text[:1500]

        court_patterns = [
            r'(?:IN THE\s+)?([A-Z][A-Za-z\s]+(?:COURT|TRIBUNAL|ARBITRATION|PANEL)[A-Za-z\s]*)',
            r'(?:BEFORE THE\s+)?([A-Z][A-Za-z\s]+(?:COURT|TRIBUNAL|ARBITRATION|PANEL)[A-Za-z\s]*)',
            r'(?:AMERICAN ARBITRATION ASSOCIATION)',
            r'(?:JAMS)',
            r'(?:ICC ARBITRATION)',
        ]

        for pattern in court_patterns:
            match = re.search(pattern, text_to_search)
            if match:
                court = match.group(0) if match.lastindex is None else match.group(1)
                return court.strip()

        return ""
