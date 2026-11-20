"""Citation extraction, syntax validation, and metadata enrichment engine (Phase 10.3).

Enforces strict programmatic validation of citations using the unambiguous format:
    [[cite:actual_chunk_id]]

Rules enforced:
1. Every substantive answer must contain at least one valid citation.
2. Validates all citations against the exact set of chunk IDs supplied to that specific request.
3. Explicitly rejects unknown, fabricated, or malformed citation IDs.
4. Explicitly rejects bracketed filenames (e.g. [CSE202.pdf]), page numbers, or doc numbers as valid citations.
5. Populates citation metadata exclusively from verified retrieved chunk metadata.
6. Enforces statement-level citation coverage so responses cannot pass by appending a single citation
   alongside ungrounded claims.
7. Explicitly distinguishes structural citation validation from semantic evidence verification.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Optional

from app.rag.retrieval import RetrievedChunk
from app.schemas.chat import CitationItem, CitationValidationResult, CitationValidationStatus

logger = logging.getLogger(__name__)

# Strict canonical citation pattern: [[cite:<chunk_id>]]
CITATION_PATTERN = re.compile(r"\[\[cite:([a-zA-Z0-9_\-\.]+)\]\]")

# Pattern to detect pseudo/malformed citations (e.g. [1], [2], [cite:...], [CSE202.pdf],
# [Source: CSE202.pdf], [Doc 1], [Page 4], [FAQ013], [[citation:...]], [[cite:]])
MALFORMED_CITATION_PATTERN = re.compile(
    r"\[\[?(?:"
    r"(?:cite|citation|chunk)\s*:[^\[\]]*"
    r"|\d+(?:\s*[-,]\s*\d+)*"
    r"|[^\[\]]*\.(?:pdf|docx|doc|json|txt|xlsx)\b[^\[\]]*"
    r"|(?:Doc|Document|Page|Section|Source|Ref|Reference|FAQ)\s*(?::|\d)[^\[\]]*"
    r")\]?\]",
    re.IGNORECASE,
)

# Abbreviations whose trailing period must not trigger a sentence split
_ABBREVIATION_PATTERN = re.compile(
    r"\b(e\.g|i\.e|vs|etc|cf|Dr|Prof|Mr|Mrs|Ms|Rs|No|Sec|Fig|Vol|Ed|Dept|Approx|p|pp|U\.S|B\.Tech|M\.Tech)\.",
    re.IGNORECASE,
)

# Leading citation tag(s) detached after a sentence-ending period (e.g. "Fact. [[cite:abc]]")
_LEADING_CITES_PATTERN = re.compile(r"^((?:\[\[cite:[a-zA-Z0-9_\-\x00]+\]\]\s*)+)")

# Pure Markdown horizontal rules or table divider lines
_MARKDOWN_DIVIDER_PATTERN = re.compile(r"^(?:[-*_]{3,}|\|[\s:\-|]+\|)$")

SEMANTIC_NOTE_VALID = (
    "Structural citation syntax, chunk-ID whitelist, and 100% statement coverage passed. "
    "Structural validation does not establish semantic correctness or factual entailment."
)
SEMANTIC_NOTE_FAILED = (
    "Structural citation validation failed; generated model output was rejected and withheld. "
    "Semantic correctness is not verified."
)
SEMANTIC_NOTE_BYPASSED = (
    "Citation validation bypassed for deterministic non-LLM response. "
    "Semantic grounding validation is not applicable."
)


@dataclass
class ValidationReport:
    """Internal evaluation report from citation validation."""

    is_valid: bool
    status: CitationValidationStatus
    valid_chunk_ids: list[str] = field(default_factory=list)
    invalid_tags: list[str] = field(default_factory=list)
    citation_items: list[CitationItem] = field(default_factory=list)
    substantive_statements_count: int = 0
    cited_statements_count: int = 0
    structural_coverage_passed: bool = False
    rejection_reason: Optional[str] = None


class CitationValidator:
    """Programmatic validator for model-generated academic citations."""

    def __init__(self):
        pass

    def extract_citations(self, text: str) -> list[str]:
        """Extracts all chunk IDs referenced via canonical [[cite:<chunk_id>]] tags."""
        if not text:
            return []
        return CITATION_PATTERN.findall(text)

    def extract_malformed_or_pseudo_citations(self, text: str) -> list[str]:
        """Detects strings resembling citations or document references that do not conform to canonical syntax."""
        if not text:
            return []
        matches = MALFORMED_CITATION_PATTERN.findall(text)
        # Filter out valid canonical citations
        malformed = []
        for m in matches:
            if not CITATION_PATTERN.fullmatch(m):
                malformed.append(m)
        return malformed

    def _protect_non_sentence_periods(self, line: str) -> str:
        """Replaces dots in abbreviations, decimals, and citation tags with sentinel \x00."""
        # Protect dots inside [[cite:...]] tags
        protected = CITATION_PATTERN.sub(
            lambda m: m.group(0).replace(".", "\x00"),
            line,
        )
        # Protect common abbreviations (including multi-dot ones like e.g. / i.e.)
        protected = _ABBREVIATION_PATTERN.sub(
            lambda m: m.group(0).replace(".", "\x00"),
            protected,
        )
        # Protect decimal numbers such as 8.50 or 75.5
        protected = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}\x00{m.group(2)}", protected)
        return protected

    def split_into_substantive_statements(self, text: str) -> list[str]:
        """Splits answer text into sentence-level substantive statements for 100% coverage checking.

        Handles abbreviations, decimals, numbered lists, Markdown formatting, and citations
        attached either before or after terminal punctuation. Does NOT exempt short factual
        statements or 'Note:' / 'Disclaimer:' / 'In summary' prefixes.
        """
        if not text or not text.strip():
            return []

        statements: list[str] = []
        raw_lines = text.replace("\r\n", "\n").strip().split("\n")

        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue
            if _MARKDOWN_DIVIDER_PATTERN.match(line):
                continue

            # Strip leading Markdown heading/bullet/numbered-list prefix without touching content
            cleaned_line = re.sub(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[\.\)]\s+)", "", line).strip()
            if not cleaned_line:
                continue

            # Skip pure structural lead-in headings ending with ':' that have no citation
            # (e.g. "The topics covered in CSE202 include:" or "**Summary:**")
            unbolded = re.sub(r"[*_`]+", "", cleaned_line).strip()
            if unbolded.endswith(":") and not CITATION_PATTERN.search(cleaned_line):
                continue

            protected = self._protect_non_sentence_periods(cleaned_line)
            raw_parts = re.split(r"(?<=[.!?])\s+", protected)

            line_sentences: list[str] = []
            for part in raw_parts:
                part = part.strip()
                if not part:
                    continue

                # If a part begins with citation tag(s) placed after the previous period
                # (e.g. "Fact one. [[cite:abc]] Fact two."), re-attach them to the prior sentence.
                leading_match = _LEADING_CITES_PATTERN.match(part)
                if leading_match and line_sentences:
                    detached_cites = leading_match.group(1).strip()
                    line_sentences[-1] = f"{line_sentences[-1]} {detached_cites}"
                    remainder = part[leading_match.end():].strip()
                    if not remainder:
                        continue
                    part = remainder

                restored = part.replace("\x00", ".").strip()
                # Ignore fragments with no alphanumeric content at all
                if not re.search(r"[a-zA-Z0-9]", restored):
                    continue
                line_sentences.append(restored)

            statements.extend(line_sentences)

        return statements

    def validate(
        self,
        answer_text: str,
        supplied_chunks: list[RetrievedChunk],
        is_abstention: bool = False,
    ) -> ValidationReport:
        """Validates citations in generated text against the exact supplied chunks.

        Enforces:
        - Non-empty answer with at least one valid canonical [[cite:chunk_id]] citation.
        - Strict whitelist check against supplied_chunks.
        - Rejection of all malformed and pseudo-citations (even alongside valid citations).
        - 100% sentence-level structural citation coverage across all substantive statements.
        """
        if is_abstention:
            return ValidationReport(
                is_valid=True,
                status=CitationValidationStatus.BYPASSED,
                structural_coverage_passed=True,
            )

        if not answer_text or not answer_text.strip():
            return ValidationReport(
                is_valid=False,
                status=CitationValidationStatus.FAILED_REJECTED,
                rejection_reason="Answer text is empty or whitespace only.",
            )

        supplied_map: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in supplied_chunks}
        supplied_ids = set(supplied_map.keys())

        # 1. Extract canonical citations
        extracted_ids = self.extract_citations(answer_text)

        # 2. Extract pseudo/malformed citations (checked even if valid canonical tags exist)
        malformed_tags = self.extract_malformed_or_pseudo_citations(answer_text)

        # 3. Check for at least one citation
        if not extracted_ids:
            return ValidationReport(
                is_valid=False,
                status=CitationValidationStatus.FAILED_REJECTED,
                valid_chunk_ids=[],
                invalid_tags=malformed_tags,
                citation_items=[],
                structural_coverage_passed=False,
                rejection_reason="No valid [[cite:chunk_id]] citations found in substantive answer.",
            )

        # 4. Whitelist verification: verify every extracted ID exists in supplied_ids
        valid_ids: list[str] = []
        invalid_tags: list[str] = list(malformed_tags)

        for cid in extracted_ids:
            if cid in supplied_ids:
                if cid not in valid_ids:
                    valid_ids.append(cid)
            else:
                invalid_tags.append(f"[[cite:{cid}]] (unknown chunk_id)")

        if invalid_tags:
            return ValidationReport(
                is_valid=False,
                status=CitationValidationStatus.FAILED_REJECTED,
                valid_chunk_ids=[],
                invalid_tags=invalid_tags,
                citation_items=[],
                structural_coverage_passed=False,
                rejection_reason=f"Answer contains {len(invalid_tags)} invalid, unknown, or malformed citation(s).",
            )

        # 5. Sentence-level 100% structural coverage check
        statements = self.split_into_substantive_statements(answer_text)
        cited_statements_count = 0

        for stmt in statements:
            stmt_citations = self.extract_citations(stmt)
            if stmt_citations and all(cid in supplied_ids for cid in stmt_citations):
                cited_statements_count += 1

        substantive_count = len(statements)
        coverage_passed = substantive_count >= 1 and cited_statements_count == substantive_count

        if not coverage_passed:
            logger.warning(
                "Citation coverage check failed: %d/%d substantive statements cited (100%% required).",
                cited_statements_count,
                substantive_count,
            )
            return ValidationReport(
                is_valid=False,
                status=CitationValidationStatus.FAILED_REJECTED,
                valid_chunk_ids=[],
                invalid_tags=[],
                citation_items=[],
                substantive_statements_count=substantive_count,
                cited_statements_count=cited_statements_count,
                structural_coverage_passed=False,
                rejection_reason=(
                    f"Insufficient citation coverage: {cited_statements_count}/{substantive_count} "
                    f"substantive statements cited (100% structural coverage required)."
                ),
            )

        # 6. Map valid citations exclusively to system metadata
        citation_items: list[CitationItem] = []
        for cid in valid_ids:
            chunk = supplied_map[cid]
            meta = chunk.metadata or {}
            citation_items.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=meta.get("source_file"),
                    category=meta.get("category"),
                    course_code=meta.get("course_code"),
                    title=meta.get("title") or (meta.get("headings", [""])[0] if meta.get("headings") else None),
                )
            )

        return ValidationReport(
            is_valid=True,
            status=CitationValidationStatus.VALID,
            valid_chunk_ids=valid_ids,
            invalid_tags=[],
            citation_items=citation_items,
            substantive_statements_count=substantive_count,
            cited_statements_count=cited_statements_count,
            structural_coverage_passed=True,
        )

    def to_schema_result(self, report: ValidationReport) -> CitationValidationResult:
        """Converts internal ValidationReport into API schema CitationValidationResult."""
        if report.status in (CitationValidationStatus.VALID, CitationValidationStatus.REGENERATED_VALID):
            note = SEMANTIC_NOTE_VALID
        elif report.status == CitationValidationStatus.BYPASSED:
            note = SEMANTIC_NOTE_BYPASSED
        else:
            note = SEMANTIC_NOTE_FAILED

        return CitationValidationResult(
            status=report.status,
            valid_citations_count=len(report.valid_chunk_ids) if report.is_valid else 0,
            invalid_citations=report.invalid_tags,
            substantive_statements_count=report.substantive_statements_count,
            cited_statements_count=report.cited_statements_count,
            structural_coverage_passed=report.structural_coverage_passed,
            semantic_grounding_note=note,
        )
