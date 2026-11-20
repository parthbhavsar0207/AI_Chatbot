"""End-to-end grounded academic RAG chat pipeline service (Phase 10.3).

Coordinates the verified pipeline sequence:
    User Query -> RetrievalService -> CrossEncoder Reranker -> EvidenceGate
    -> Decision-Conditioned Branching -> Prompt Construction -> OllamaLLMService
    -> Programmatic Citation Validation (with 1-time controlled regeneration)
    -> Final Verified Response
"""

import logging
import re
import time
from typing import Optional

from app.rag.citation_validator import (
    SEMANTIC_NOTE_BYPASSED,
    CitationValidator,
)
from app.rag.evidence_gate import (
    SOURCE_AUTHORITY_MAP,
    SOURCE_PROVENANCE_REGISTRY,
    EvidenceDecision,
    EvidenceDecisionState,
    is_chunk_verified_for_generation,
)
from app.rag.retrieval import (
    RetrievalService,
    RetrievedChunk,
    get_last_retrieval_timings,
    reset_last_retrieval_timings,
)
from app.schemas.chat import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    CitationItem,
    CitationValidationResult,
    CitationValidationStatus,
    SourceMetadataItem,
)
from app.services.llm_service import BaseLLMService, OllamaLLMService, get_llm_service
from app.services.retrieval_service import get_retrieval_service

logger = logging.getLogger(__name__)

# Minimal deterministic whole-utterance token sets and patterns for conversational intents
# Short tokens: strictly exact-match only after repeated-character collapse (no edit distance allowed)
_SHORT_GREETING_TOKENS = frozenset({"hi", "hey"})
_SHORT_FAREWELL_TOKENS = frozenset({"bye"})

# Longer tokens: support bounded 1-char edit distance and repeated-character collapse
_LONGER_CANONICAL_TOKENS = frozenset({
    "hello",
    "good",
    "morning",
    "afternoon",
    "evening",
    "thanks",
    "thank",
    "you",
    "goodbye",
    "farewell",
})

# Casual addressees supported in conversational utterances
_CASUAL_ADDRESSEES = frozenset({
    "buddy",
    "friend",
    "bro",
    "mate",
    "pal",
    "team",
    "there",
    "bot",
    "assistant",
    "all",
    "everyone",
})

# Conversational modifiers and filler words
_CONVERSATIONAL_FILLERS = frozenset({
    "very",
    "so",
    "much",
    "a",
    "lot",
    "for",
    "now",
    "see",
    "later",
    "again",
    "to",
})

# Academic guard keywords: presence of any of these immediately disqualifies the query from conversational intent
_ACADEMIC_GUARD_KEYWORDS = frozenset({
    "cse", "int", "mth", "pel", "che", "phy", "ece", "mec",
    "ai", "ml", "aiml", "syllabus", "course", "courses", "subject", "subjects",
    "exam", "exams", "examination", "credit", "credits", "prerequisite", "prerequisites",
    "attendance", "unit", "units", "inheritance", "algorithm", "textbook", "textbooks",
    "dean", "curriculum", "grade", "grades", "cgpa", "sgpa", "degree", "fee", "fees",
    "placement", "placements", "class", "classes", "lecture", "lectures", "lab", "labs",
    "assignment", "assignments", "test", "tests", "ca", "term", "semester", "topic", "topics",
    "passing", "marks", "grading", "policy", "rules", "regulation", "regulations",
})

# Question / imperative words indicating an academic or informational query
_QUESTION_INSTRUCTION_WORDS = frozenset({
    "what", "when", "where", "why", "how", "which", "who", "whose", "whom",
    "explain", "describe", "define", "detail", "details", "tell", "show", "list", "give",
})

_COURSE_CODE_PATTERN = re.compile(r"\b[a-zA-Z]{2,4}\s*\d{3}\b")

# Conversational prefix pattern for stripping from academic queries before retrieval/reranking.
# Strips recognized conversational prefixes only when followed by an explicit punctuation delimiter
# (comma, period, exclamation mark, question mark, colon, semicolon, hyphen, tilde).
# Bare whitespace never independently triggers prefix stripping.
_CONVERSATIONAL_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"hi|hello|hey|greetings|"
    r"good\s+(?:morning|afternoon|evening)|"
    r"thanks|thank\s+you|"
    r"bye|goodbye|good\s+bye"
    r")"
    r"(?:\s+(?:there|bot|assistant|buddy|friend|bro|mate|pal|team|all|everyone))?"
    r"\s*[,.!?:;\-~]+\s*",
    re.IGNORECASE,
)

# Conceptual explanation query pattern
_CONCEPTUAL_EXPLANATION_PATTERN = re.compile(
    r"\b(?:explain|what\s+is|define|how\s+does|how\s+do|teach\s+me|difference\s+between|concept\s+of|meaning\s+of|describe)\b",
    re.IGNORECASE,
)

_CONCEPTUAL_STOPWORDS = frozenset({
    "explain", "what", "is", "are", "the", "a", "an", "and", "or", "in", "of",
    "for", "to", "how", "does", "do", "work", "difference", "between", "concept",
    "meaning", "describe", "define", "teach", "me", "about", "please", "tell",
    "with", "by", "its", "can", "you", "give", "examples", "example",
})


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
        prev = curr
    return prev[-1]


def _normalize_conversational_token(tok: str) -> str:
    """Normalizes a single token by collapsing repeated characters and bounded 1-char edit distance.

    Short tokens ('hi', 'hey', 'bye') strictly require exact match (after repeat collapse).
    Longer tokens support bounded edit distance <= 1.
    """
    tok = tok.lower()
    # 1. Exact match check
    if (
        tok in _SHORT_GREETING_TOKENS
        or tok in _SHORT_FAREWELL_TOKENS
        or tok in _LONGER_CANONICAL_TOKENS
        or tok in _CASUAL_ADDRESSEES
        or tok in _CONVERSATIONAL_FILLERS
    ):
        return tok

    # 2. Collapse repeated characters
    c_single = re.sub(r"(.)\1+", r"\1", tok)
    c_double = re.sub(r"(.)\1{2,}", r"\1\1", tok)

    # Short tokens: exact match only after repeat collapse (prevent collision with 'ai', 'by')
    if c_single in _SHORT_GREETING_TOKENS:
        return c_single
    if c_single in _SHORT_FAREWELL_TOKENS:
        return c_single

    # Longer tokens / addressees / fillers exact match after repeat collapse
    for cand in (c_double, c_single):
        if cand in _LONGER_CANONICAL_TOKENS:
            return cand
        if cand in _CASUAL_ADDRESSEES:
            return cand
        if cand in _CONVERSATIONAL_FILLERS:
            return cand

    # 3. Bounded edit distance <= 1 for longer canonical tokens (len >= 4)
    for target in _LONGER_CANONICAL_TOKENS:
        if len(target) >= 4:
            for cand in (tok, c_double, c_single):
                if _levenshtein_distance(cand, target) <= 1:
                    return target

    for target in _CASUAL_ADDRESSEES:
        if len(target) >= 4:
            for cand in (tok, c_double, c_single):
                if _levenshtein_distance(cand, target) <= 1:
                    return target

    return ""


def detect_conversational_intent(query: str) -> Optional[tuple[str, str]]:
    """Detects standalone conversational intents (greeting, gratitude, farewell).

    Guarantees:
    - Normalizes repeated character sequences ('goood' -> 'good', 'heyyyy' -> 'hey').
    - Supports bounded 1-char edit distance for longer tokens ('mornig' -> 'morning').
    - Keeps short tokens ('hi', 'hey', 'bye') exact-match only to prevent collision with 'ai', 'by', etc.
    - Supports casual addressees ('buddy', 'friend', 'bro', 'mate', etc.).
    - Strict whole-utterance matching: unrecognized tokens cause fail-closed return None.
    - Enforces academic-query guard: academic questions ('Hi, what is CSE202?', 'AI and ML')
      immediately return None and bypass conversational intent.

    Returns:
        Optional[tuple[str, str]]: (intent_name, deterministic_reply) or None.
    """
    if not query or not query.strip():
        return None

    raw = query.strip()

    # Academic-query guard: reject immediately if academic terms, question words, or course codes exist
    tokens_raw = set(re.findall(r"\b[a-zA-Z]+\b", raw.lower()))
    if tokens_raw & _ACADEMIC_GUARD_KEYWORDS:
        return None
    if tokens_raw & _QUESTION_INSTRUCTION_WORDS:
        return None
    if _COURSE_CODE_PATTERN.search(raw):
        return None

    words = re.findall(r"[a-zA-Z]+", raw)
    if not words:
        return None

    normalized_words = [_normalize_conversational_token(w) for w in words]
    if any(not nw for nw in normalized_words):
        # Contains unrecognized token -> fail closed
        return None

    phrase = " ".join(normalized_words)

    # Greeting pattern
    greeting_re = re.compile(
        r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))"
        r"(?:\s+(?:buddy|friend|bro|mate|pal|team|there|bot|assistant|all|everyone))?$"
    )
    if greeting_re.fullmatch(phrase):
        if phrase.startswith("good "):
            time_of_day = phrase.split()[1].lower()
            salutation = f"Good {time_of_day}!"
        else:
            salutation = "Hello!"
        return (
            "greeting",
            f"{salutation} I am the LPU CSE-AIML Student Assistance chatbot. "
            "Ask me about indexed course syllabi, course topics, textbooks, or academic guidelines.",
        )

    # Gratitude pattern
    gratitude_re = re.compile(
        r"^(?:thanks|thank\s+you)"
        r"(?:\s+(?:very\s+much|so\s+much|a\s+lot))?"
        r"(?:\s+(?:buddy|friend|bro|mate|pal|team|there|bot|assistant))?$"
    )
    if gratitude_re.fullmatch(phrase):
        return (
            "gratitude",
            "You're welcome! Let me know if you have any other questions about your "
            "LPU CSE-AIML courses, syllabi, or academic guidelines.",
        )

    # Farewell pattern
    farewell_re = re.compile(
        r"^(?:bye|goodbye)"
        r"(?:\s+(?:for\s+now|see\s+you\s+later))?"
        r"(?:\s+(?:buddy|friend|bro|mate|pal|team|there|bot|assistant))?$"
    )
    if farewell_re.fullmatch(phrase):
        return (
            "farewell",
            "Goodbye! Feel free to return anytime if you have questions about your "
            "LPU CSE-AIML courses or academic guidelines.",
        )

    return None


def strip_conversational_prefix(query: str) -> str:
    """Strips leading conversational greeting/gratitude/farewell prefix from academic queries."""
    if not query:
        return query
    match = _CONVERSATIONAL_PREFIX_PATTERN.match(query.strip())
    if match:
        remainder = query.strip()[match.end():].strip()
        if remainder:
            return remainder
    return query.strip()


def _check_conceptual_clarification(query: str, retrieved_chunks: list[RetrievedChunk]) -> Optional[str]:
    """Detects if an INSUFFICIENT query is a conceptual request whose topic appears in a verified syllabus.

    Returns a deterministic clarification that the indexed syllabus lists course topics but lacks
    instructional detail, without fabricating explanations or citations.
    """
    if not _CONCEPTUAL_EXPLANATION_PATTERN.search(query):
        return None

    tokens = re.findall(r"[a-zA-Z0-9#+]+", query.lower())
    candidate_topics = [
        t for t in tokens
        if t not in _CONCEPTUAL_STOPWORDS and not re.fullmatch(r"[a-z]{2,4}\d{3}", t) and len(t) >= 3
    ]
    if not candidate_topics:
        return None

    verified_chunks = [c for c in retrieved_chunks if is_chunk_verified_for_generation(c)]
    if not verified_chunks:
        return None

    for c in verified_chunks[:3]:
        meta = c.metadata or {}
        category = meta.get("category", "")
        source_file = meta.get("source_file", "")
        is_syllabus = category == "syllabus" or source_file.endswith(".pdf")
        if not is_syllabus:
            continue
        text_lower = c.text.lower()
        for topic in candidate_topics:
            if re.search(rf"\b{re.escape(topic)}\b", text_lower):
                course_code = meta.get("course_code") or (source_file.replace(".pdf", "") if source_file else "the course")
                return (
                    f"The indexed official syllabus for {course_code} ({source_file}) lists '{topic.title()}' "
                    "as a curriculum topic, but university syllabus outlines do not contain instructional textbook "
                    "explanations or pedagogical definitions. Because the indexed corpus covers syllabus curriculum outlines "
                    "rather than instructional textbooks, please consult the recommended course textbooks or lecture materials "
                    "for a detailed conceptual explanation."
                )
    return None

SYSTEM_PROMPT_ACADEMIC_OLLAMA = (
    "You are an official academic assistant for LPU students. "
    "Answer the student's question factually, concisely, and strictly based on the provided context chunks. "
    "MANDATORY CITATION RULE: Every single factual sentence or bullet item must include its "
    "exact citation tag in the format: [[cite:chunk_id]]. "
    "Do NOT use document filenames, page numbers, FAQ labels, or generic numbers like [1] as citations. "
    "Use ONLY the exact chunk_id strings provided in the context headers. "
    "Do NOT speculate, assume, or introduce information not explicitly stated in the context."
)

SYSTEM_PROMPT_ACADEMIC_GEMINI = (
    "You are an official academic assistant for LPU students. "
    "Answer the student's question factually, clearly, and strictly based on the provided context chunks.\n\n"
    "RESPONSE AND FORMATTING GUIDELINES:\n"
    "1. Start with a direct answer to the user's question. Keep simple answers short and focused.\n"
    "2. Use clear, natural, and student-friendly language. Avoid repetitive introductions, unnecessary conclusions, and generic filler.\n"
    "3. For longer answers, organize content with descriptive Markdown headings and organized bullet points instead of monolithic paragraphs.\n"
    "4. Group related topics logically into distinct sections.\n"
    "5. Use numbered steps for procedures, rules, or explanations involving sequences.\n"
    "6. Present structured dates, schedules, or comparisons clearly using bullet lists or structured Markdown.\n"
    "7. Highlight important terms sparingly using bold text.\n"
    "8. Never speculate, assume, or expand beyond the retrieved evidence. If complete code implementations or details are not in the context, explain factually what the syllabus/documents provide without fabricating ungrounded examples.\n\n"
    "MANDATORY CITATION RULES (STRICT):\n"
    "- Every single factual sentence, bullet point, or statement MUST include its exact citation tag in the format: [[cite:chunk_id]].\n"
    "- Place the citation tag immediately next to the claim it supports.\n"
    "- All Markdown section headings MUST end with a colon ':' (e.g. '### Examination Schedule:').\n"
    "- If using a Markdown table, the header row must end with a colon (e.g. '| Course | Date | Time :|') or contain a citation.\n"
    "- Use ONLY the exact chunk_id strings provided in the context headers. Do NOT use filenames (e.g. [CSE202.pdf]), page numbers, or generic numbers like [1].\n"
    "- Do NOT fabricate citations or remove citations to improve appearance."
)

SYSTEM_PROMPT_ACADEMIC = SYSTEM_PROMPT_ACADEMIC_GEMINI


class RAGChatService:
    """Orchestrates decision-conditioned RAG generation with programmatic citation enforcement."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_service: Optional[BaseLLMService] = None,
        citation_validator: Optional[CitationValidator] = None,
    ):
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.llm_service = llm_service or get_llm_service()
        self.citation_validator = citation_validator or CitationValidator()

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """Formats retrieved chunks into unambiguous, identifiable context blocks."""
        blocks = []
        for c in chunks:
            meta = c.metadata or {}
            source_file = meta.get("source_file", "unknown")
            category = meta.get("category", "general")
            header = f"--- CHUNK ID: {c.chunk_id} | Source: {source_file} | Category: {category} ---"
            blocks.append(f"{header}\n{c.text.strip()}\n")
        return "\n".join(blocks)

    def _build_citations_from_chunks(self, chunks: list[RetrievedChunk]) -> list[CitationItem]:
        """Builds CitationItem objects exclusively from RetrievedChunk metadata."""
        items: list[CitationItem] = []
        seen_ids: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            meta = chunk.metadata or {}
            items.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=meta.get("source_file"),
                    category=meta.get("category"),
                    course_code=meta.get("course_code"),
                    title=meta.get("title") or (meta.get("headings", [""])[0] if meta.get("headings") else None),
                )
            )
        return items

    def _build_source_metadata(self, citation_items: list[CitationItem]) -> list[SourceMetadataItem]:
        """Enriches valid citations with authentic provenance metadata."""
        seen_files = set()
        metadata_items = []
        for item in citation_items:
            fname = item.source_file
            if not fname or fname in seen_files:
                continue
            seen_files.add(fname)
            prov = SOURCE_PROVENANCE_REGISTRY.get(fname, {})
            auth = SOURCE_AUTHORITY_MAP.get(fname, prov.get("authority_level", "unknown"))
            metadata_items.append(
                SourceMetadataItem(
                    source_file=fname,
                    authority_level=auth,
                    document_origin=prov.get("document_origin"),
                    institutional_authenticity=prov.get("institutional_authenticity"),
                    verification_status=prov.get("verification_status"),
                )
            )
        return metadata_items

    async def answer_question(self, request: ChatAnswerRequest) -> ChatAnswerResponse:
        """Executes the complete decision-conditioned RAG generation pipeline."""
        t_start = time.perf_counter()

        # -------------------------------------------------------------------
        # 0. Deterministic Conversational Intent Short-Circuit (Bypass RAG & LLM)
        # -------------------------------------------------------------------
        conversational = detect_conversational_intent(request.query)
        if conversational is not None:
            intent_name, reply_text = conversational
            logger.info("Detected standalone conversational intent '%s'; bypassing retrieval and LLM.", intent_name)
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            return ChatAnswerResponse(
                query=request.query,
                answer=reply_text,
                evidence_state=EvidenceDecisionState.SUFFICIENT,
                citations=[],
                source_metadata=[],
                missing_information=None,
                conflict_details=None,
                model_identifier="deterministic_conversational_handler",
                retrieval_latency_ms=0.0,
                reranking_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_ms,
                citation_validation=CitationValidationResult(
                    status=CitationValidationStatus.BYPASSED,
                    valid_citations_count=0,
                    structural_coverage_passed=True,
                    semantic_grounding_note=SEMANTIC_NOTE_BYPASSED,
                ),
                citation_validation_status=CitationValidationStatus.BYPASSED,
            )

        # Normalize academic query by stripping recognized conversational prefixes
        rag_query = strip_conversational_prefix(request.query)

        # -------------------------------------------------------------------
        # 1. Retrieval & Reranking (independently timed, reset on exit/error)
        # -------------------------------------------------------------------
        reset_last_retrieval_timings()
        t_retrieval_start = time.perf_counter()
        try:
            retrieved_chunks = self.retrieval_service.retrieve(
                query=rag_query,
                top_k=request.top_k,
                category=request.category,
                course_code=request.course_code,
                source_file=request.source_file,
                enable_rerank=True,
            )
            stage_timings = get_last_retrieval_timings()
        finally:
            reset_last_retrieval_timings()
        t_retrieval_end = time.perf_counter()

        if stage_timings is not None:
            retrieval_latency_ms, reranking_latency_ms = stage_timings
        else:
            # Fallback when RetrievalService.retrieve is mocked in unit tests
            retrieval_latency_ms = round((t_retrieval_end - t_retrieval_start) * 1000.0, 2)
            reranking_latency_ms = 0.0

        # -------------------------------------------------------------------
        # 2. Evidence Sufficiency Evaluation
        # -------------------------------------------------------------------
        decision: EvidenceDecision = self.retrieval_service.evidence_gate.evaluate(
            query=rag_query,
            retrieved_chunks=retrieved_chunks,
        )

        state = decision.decision
        missing_info = decision.missing_information
        conflict_details = decision.conflict_details

        # -------------------------------------------------------------------
        # 3. Decision-Conditioned Branching
        # -------------------------------------------------------------------

        # Branch A: INSUFFICIENT -> Deterministic Abstention (Bypass LLM)
        if state == EvidenceDecisionState.INSUFFICIENT:
            logger.info("EvidenceGate state INSUFFICIENT; bypassing LLM generation.")
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            # Check if query asks for a conceptual explanation listed in a verified syllabus
            clarification = _check_conceptual_clarification(rag_query, retrieved_chunks)
            if clarification:
                abstention_text = clarification
            else:
                abstention_text = (
                    "The available university documents do not sufficiently support an answer to this question. "
                    "The requested topic is either outside the indexed official documents or lacks necessary operational detail. "
                    "Please consult the relevant university department, official portal, or examination board directly."
                )
            return ChatAnswerResponse(
                query=request.query,
                answer=abstention_text,
                evidence_state=state,
                citations=[],
                source_metadata=[],
                missing_information=missing_info,
                conflict_details=None,
                model_identifier="deterministic_abstention_engine",
                retrieval_latency_ms=retrieval_latency_ms,
                reranking_latency_ms=reranking_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_ms,
                citation_validation=CitationValidationResult(
                    status=CitationValidationStatus.BYPASSED,
                    valid_citations_count=0,
                    structural_coverage_passed=True,
                    semantic_grounding_note=SEMANTIC_NOTE_BYPASSED,
                ),
                citation_validation_status=CitationValidationStatus.BYPASSED,
            )

        # Branch B: CONFLICTING -> Deterministic Conflict/Caveat Summary (Bypass LLM)
        if state == EvidenceDecisionState.CONFLICTING:
            logger.info("EvidenceGate state CONFLICTING; bypassing LLM generation.")
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            details = dict(conflict_details or {})
            classification = details.get("conflict_classification", "untraceable_conflict")
            is_synthetic = bool(details.get("is_synthetic", False)) or classification == "synthetic_benchmark"

            # Resolve traceable chunks strictly from actual retrieved_chunks
            chunk_map = {c.chunk_id: c for c in retrieved_chunks}
            traceable_chunks: list[RetrievedChunk] = []
            if not is_synthetic:
                ref_ids = [
                    ref["chunk_id"]
                    for ref in details.get("source_references", [])
                    if isinstance(ref, dict) and ref.get("chunk_id") in chunk_map
                ]
                if not ref_ids:
                    ref_ids = [
                        cid
                        for cid in details.get("affected_chunk_ids", [])
                        if cid in chunk_map
                    ]
                traceable_chunks = [chunk_map[cid] for cid in dict.fromkeys(ref_ids)]

            conflict_citations = self._build_citations_from_chunks(traceable_chunks)
            conflict_source_meta = self._build_source_metadata(conflict_citations)

            def _fmt_source_ref(ref: dict) -> str:
                fname = ref.get("source_file", "unknown")
                auth = ref.get("authority_level") or SOURCE_AUTHORITY_MAP.get(fname, "unclassified")
                ver = ref.get("verification_status") or "unknown"
                cid = ref.get("chunk_id", "")
                return f"{fname} (authority: {auth}; verification: {ver}) [[cite:{cid}]]"

            source_refs = [
                r for r in details.get("source_references", [])
                if isinstance(r, dict) and r.get("chunk_id") in chunk_map
            ]

            if classification == "actual_conflict" and len(source_refs) >= 2:
                refs_summary = "; ".join(_fmt_source_ref(r) for r in source_refs)
                conflict_text = (
                    f"Incompatible competing policy requirements were detected across retrieved sources: {refs_summary}. "
                    "Because competing institutional policy claims cannot be automatically adjudicated by an AI assistant, "
                    "please verify the currently effective requirement with the Academic Dean, Registrar's Office, or official student portal."
                )
            elif classification == "verification_caveat" and len(source_refs) >= 2:
                req_refs = [r for r in source_refs if r.get("role") == "stated_requirement"]
                cav_refs = [r for r in source_refs if r.get("role") == "verification_caveat"]
                req_desc = "; ".join(_fmt_source_ref(r) for r in req_refs) or "retrieved requirement source"
                cav_desc = "; ".join(_fmt_source_ref(r) for r in cav_refs) or "retrieved verification caveat source"
                conflict_text = (
                    f"A policy verification caveat was detected across retrieved sources (not confirmed competing official policies). "
                    f"Stated requirement reference: {req_desc}. "
                    f"Verification caveat reference: {cav_desc}. "
                    "Please confirm the currently enforceable rule with the Academic Dean, Registrar's Office, or official student portal."
                )
            else:
                conflict_text = (
                    "A potential policy contradiction was detected, but competing policy claims could not be traced "
                    "to the currently retrieved evidence. "
                    "Please verify the effective policy directly with the Academic Dean, Registrar's Office, or official student portal."
                )

            return ChatAnswerResponse(
                query=request.query,
                answer=conflict_text,
                evidence_state=state,
                citations=conflict_citations,
                source_metadata=conflict_source_meta,
                missing_information=missing_info,
                conflict_details=details or {"conflict_classification": "untraceable_conflict", "explanation": decision.explanation},
                model_identifier="deterministic_conflict_engine",
                retrieval_latency_ms=retrieval_latency_ms,
                reranking_latency_ms=reranking_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_ms,
                citation_validation=CitationValidationResult(
                    status=CitationValidationStatus.BYPASSED,
                    valid_citations_count=0,
                    structural_coverage_passed=True,
                    semantic_grounding_note=SEMANTIC_NOTE_BYPASSED,
                ),
                citation_validation_status=CitationValidationStatus.BYPASSED,
            )

        # Branch C & D: PARTIAL and SUFFICIENT -> Controlled LLM Generation
        # Strict fail-closed filtering for PARTIAL state (no arbitrary retrieved_chunks[:3] fallback)
        if state == EvidenceDecisionState.PARTIAL:
            supp_ids = set(decision.supporting_chunk_ids or [])
            evidence_chunks = [
                c
                for c in retrieved_chunks
                if c.chunk_id in supp_ids and is_chunk_verified_for_generation(c)
            ]
            if not evidence_chunks:
                logger.info(
                    "PARTIAL state has zero verified supporting chunks; returning deterministic abstention without LLM."
                )
                total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                abstention_text = (
                    "The available retrieved documents do not contain verified supporting evidence sufficient "
                    "to answer this question reliably. The retrieved items are either unverified, incomplete, "
                    "or do not directly support the requested aspects. Please consult the official university "
                    "student portal, academic handbook, or relevant department directly."
                )
                return ChatAnswerResponse(
                    query=request.query,
                    answer=abstention_text,
                    evidence_state=state,
                    citations=[],
                    source_metadata=[],
                    missing_information=missing_info,
                    conflict_details=None,
                    model_identifier="deterministic_abstention_engine",
                    retrieval_latency_ms=retrieval_latency_ms,
                    reranking_latency_ms=reranking_latency_ms,
                    generation_latency_ms=0.0,
                    total_latency_ms=total_ms,
                    citation_validation=CitationValidationResult(
                        status=CitationValidationStatus.BYPASSED,
                        valid_citations_count=0,
                        structural_coverage_passed=True,
                        semantic_grounding_note=SEMANTIC_NOTE_BYPASSED,
                    ),
                    citation_validation_status=CitationValidationStatus.BYPASSED,
                )
        else:
            evidence_chunks = retrieved_chunks[: request.top_k]

        context_block = self._build_context_block(evidence_chunks)

        # Construct Provider-Aware Prompt and System Prompt
        is_gemini = getattr(self.llm_service, "primary_model", "").startswith("gemini") or "gemini" in type(self.llm_service).__name__.lower()
        effective_system_prompt = SYSTEM_PROMPT_ACADEMIC_GEMINI if is_gemini else SYSTEM_PROMPT_ACADEMIC_OLLAMA

        if is_gemini:
            if state == EvidenceDecisionState.PARTIAL and missing_info:
                missing_str = "\n".join(f"- {m}" for m in missing_info)
                user_prompt = (
                    f"Context Evidence:\n{context_block}\n\n"
                    f"Question:\n{request.query}\n\n"
                    f"IMPORTANT NOTICE ON MISSING INFORMATION:\n"
                    f"The following aspects of the user's question are NOT supported by the university documents:\n"
                    f"{missing_str}\n\n"
                    f"Instructions:\n"
                    f"1. Start with a direct answer to the supported aspects.\n"
                    f"2. Explicitly state what information regarding missing aspects is not available in the documents in a concise and helpful way.\n"
                    f"3. Organize your answer with clear Markdown headings, bullet points, or tables as appropriate.\n"
                    f"4. Do NOT guess or infer missing rules.\n"
                    f"5. Attach [[cite:chunk_id]] to EVERY factual sentence or bullet item using only the provided chunk IDs."
                )
            else:
                user_prompt = (
                    f"Context Evidence:\n{context_block}\n\n"
                    f"Question:\n{request.query}\n\n"
                    f"Instructions:\n"
                    f"Answer the student's question directly, clearly, and factually based strictly on the context above.\n"
                    f"- Section headings must end with a colon ':' (e.g. '### Overview:').\n"
                    f"- Every single sentence or bullet item must cite its source chunk using [[cite:chunk_id]].\n"
                    f"- Do not speculate or invent details/code not present in the evidence."
                )
        else:
            if state == EvidenceDecisionState.PARTIAL and missing_info:
                missing_str = "\n".join(f"- {m}" for m in missing_info)
                user_prompt = (
                    f"Context Evidence:\n{context_block}\n\n"
                    f"Question:\n{request.query}\n\n"
                    f"IMPORTANT NOTICE ON MISSING INFORMATION:\n"
                    f"The following aspects of the user's question are NOT supported by the university documents:\n"
                    f"{missing_str}\n\n"
                    f"Instructions:\n"
                    f"1. Answer ONLY the aspects that are directly supported by the context.\n"
                    f"2. Explicitly state that information regarding the missing aspects is not available in the documents.\n"
                    f"3. Do NOT guess or infer the missing rules.\n"
                    f"4. Attach [[cite:chunk_id]] to EVERY sentence using only the provided chunk IDs."
                )
            else:
                user_prompt = (
                    f"Context Evidence:\n{context_block}\n\n"
                    f"Question:\n{request.query}\n\n"
                    f"Instructions:\n"
                    f"Answer the question directly and factually based only on the context above. "
                    f"Every single sentence must cite its source chunk using [[cite:chunk_id]]."
                )

        # -------------------------------------------------------------------
        # 4. LLM Generation & Citation Validation (with 1-time regeneration)
        # -------------------------------------------------------------------
        t_gen_start = time.perf_counter()
        raw_answer, gen_metrics = await self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=effective_system_prompt,
        )
        gen_latency_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)

        # Validate Citations
        report = self.citation_validator.validate(raw_answer, evidence_chunks)
        final_answer = raw_answer
        final_status = report.status

        # If citation validation failed on first attempt, execute AT MOST ONE controlled regeneration
        if not report.is_valid:
            logger.warning(
                "Initial generation failed citation validation (%s). Attempting 1-time controlled regeneration.",
                report.rejection_reason,
            )
            valid_ids_str = ", ".join(c.chunk_id for c in evidence_chunks)
            if is_gemini:
                regen_prompt = (
                    f"{user_prompt}\n\n"
                    f"CORRECTION REQUIREMENT: Your previous response failed citation verification because: {report.rejection_reason}.\n"
                    f"Re-answer the query factually and ensure EVERY sentence and bullet item includes [[cite:chunk_id]]. "
                    f"Ensure all section headings end with a colon ':'. "
                    f"Do NOT include numeric brackets like [1], filenames, or FAQ labels. "
                    f"Use ONLY these valid chunk IDs: {valid_ids_str}."
                )
            else:
                regen_prompt = (
                    f"{user_prompt}\n\n"
                    f"CORRECTION REQUIREMENT: Your previous response failed citation verification because: {report.rejection_reason}.\n"
                    f"Re-answer the query factually and ensure EVERY sentence includes [[cite:chunk_id]]. "
                    f"Do NOT include numeric brackets like [1], filenames, or FAQ labels. "
                    f"Use ONLY these valid chunk IDs: {valid_ids_str}."
                )
            t_regen_start = time.perf_counter()
            regen_answer, regen_metrics = await self.llm_service.generate(
                prompt=regen_prompt,
                system_prompt=effective_system_prompt,
            )
            gen_latency_ms += round((time.perf_counter() - t_regen_start) * 1000.0, 2)

            regen_report = self.citation_validator.validate(regen_answer, evidence_chunks)
            if regen_report.is_valid:
                logger.info("Controlled regeneration passed citation validation.")
                regen_report.status = CitationValidationStatus.REGENERATED_VALID
                final_answer = regen_answer
                report = regen_report
                final_status = CitationValidationStatus.REGENERATED_VALID
            else:
                logger.error("Controlled regeneration still failed citation validation: %s", regen_report.rejection_reason)
                regen_report.status = CitationValidationStatus.FAILED_REJECTED
                regen_report.valid_chunk_ids = []
                regen_report.citation_items = []
                regen_report.structural_coverage_passed = False
                # Sanitize invalid_tags in API response so rejected model output never leaks
                if regen_report.invalid_tags:
                    regen_report.invalid_tags = [
                        f"rejected_citation_{idx + 1}"
                        for idx in range(len(regen_report.invalid_tags))
                    ]
                report = regen_report
                final_status = CitationValidationStatus.FAILED_REJECTED
                # Safe fallback to avoid delivering ungrounded claims
                final_answer = (
                    "The system retrieved relevant official documents, but the generated answer could not satisfy "
                    "strict citation verification standards. Please consult the retrieved documents directly."
                )

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        source_meta = self._build_source_metadata(report.citation_items)

        return ChatAnswerResponse(
            query=request.query,
            answer=final_answer,
            evidence_state=state,
            citations=report.citation_items,
            source_metadata=source_meta,
            missing_information=missing_info,
            conflict_details=conflict_details,
            model_identifier=gen_metrics.get("model", self.llm_service.primary_model),
            retrieval_latency_ms=retrieval_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
            generation_latency_ms=gen_latency_ms,
            total_latency_ms=total_ms,
            citation_validation=self.citation_validator.to_schema_result(report),
            citation_validation_status=final_status,
        )


def get_rag_chat_service() -> RAGChatService:
    """FastAPI dependency provider for RAGChatService."""
    return RAGChatService()
