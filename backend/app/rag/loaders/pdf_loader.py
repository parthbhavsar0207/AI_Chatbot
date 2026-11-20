"""PDF document loader with page tracking, footer pruning, and structural metadata extraction."""

import logging
from pathlib import Path
import re
from typing import Optional
from pypdf import PdfReader

from app.rag.cleaner import clean_text
from app.rag.loaders.base import BaseLoader
from app.rag.schema import Document

logger = logging.getLogger(__name__)

# Common syllabus/academic heading patterns
HEADING_REGEX = re.compile(
    r"^(COURSE OUTCOMES|COURSE OBJECTIVES|SYLLABUS|UNIT\s+[IVXLCDM\d]+|TEXT\s*BOOKS?|REFERENCES?|"
    r"SCHEME\s+OF\s+EVALUATION|EVALUATION\s+CRITERIA|PRACTICAL|LAB|PREREQUISITES?|CREDITS?)\b",
    re.IGNORECASE,
)
COURSE_CODE_REGEX = re.compile(r"\b([A-Z]{3}\d{3})\b")

# Generic/placeholder PDF titles to reject
GENERIC_PDF_TITLES = {
    "rdlcoursesyllabusnew",
    "report",
    "untitled",
    "document",
    "pdf",
    "microsoft word",
    "print",
}


def is_footer_only_page(text: str) -> bool:
    """Detects if a page contains only pagination, session indicators, and footer labels."""
    if not text or not text.strip():
        return True
    # Strip known footer elements like 'Session 2026-27' and 'Page: 2 / 2'
    filtered = re.sub(r"(?i)\bsession\s*\d{2,4}\s*[-–]?\s*\d{0,4}\b", "", text)
    filtered = re.sub(r"(?i)\bpage\s*:\s*\d+(\s*/\s*\d+)?\b", "", filtered)
    # Strip remaining digits, whitespace, and punctuation
    residual = re.sub(r"[\s\d\-_/:|.]+", "", filtered)
    return len(residual) == 0


def derive_pdf_title(first_page_text: str, course_code: Optional[str], default_title: str) -> str:
    """Extracts a meaningful course title from the first page text content."""
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    if not lines:
        return default_title

    for i, line in enumerate(lines[:6]):
        if course_code and course_code in line:
            # Title on same line after colon (e.g. 'CSE202:OBJECT ORIENTED PROGRAMMING')
            if ":" in line and len(line) > len(course_code) + 2:
                return line.strip()
            # Title on subsequent non-empty line (e.g. 'CSE205' -> ':' -> 'DATA STRUCTURES')
            subsequent = [l for l in lines[i + 1 : i + 4] if l != ":"]
            if subsequent:
                return f"{course_code}: {subsequent[0]}"
            return line.strip()

    # Fallback to first non-numeric descriptive line
    for line in lines[:4]:
        if len(line) > 5 and not line.isdigit() and not line.lower().startswith("session"):
            return line

    return default_title


class PDFLoader(BaseLoader):
    """Extracts text from PDF documents page-by-page while preserving structural metadata."""

    supported_extensions: set[str] = {"pdf"}

    def load(self, file_path: Path, category: Optional[str] = None) -> list[Document]:
        """Loads a PDF document and returns one Document per non-empty, substantive page."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning(f"PDF file has 0 pages: {path.name}")
            return []

        # Derive initial course code from filename
        course_code_match = COURSE_CODE_REGEX.search(path.stem)
        course_code = course_code_match.group(1) if course_code_match else None

        # Check metadata title, rejecting generic placeholders
        valid_meta_title = None
        if reader.metadata and reader.metadata.title:
            extracted_meta_title = reader.metadata.title.strip()
            if extracted_meta_title and extracted_meta_title.lower() not in GENERIC_PDF_TITLES:
                valid_meta_title = extracted_meta_title

        # Derive title from first page content if metadata title is missing or placeholder
        default_title = path.stem.replace("_", " ").strip()
        if valid_meta_title:
            doc_title = valid_meta_title
        elif total_pages > 0:
            first_raw = reader.pages[0].extract_text() or ""
            doc_title = derive_pdf_title(clean_text(first_raw), course_code, default_title)
        else:
            doc_title = default_title

        doc_category = category or "syllabus"
        documents: list[Document] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)

            # Exclude empty pages and non-substantive footer-only pages
            if not cleaned or is_footer_only_page(cleaned):
                logger.debug(f"Skipping non-substantive page {page_num} in {path.name}")
                continue

            # Detect headings on current page
            page_headings: list[str] = []
            for line in cleaned.split("\n"):
                stripped = line.strip()
                if HEADING_REGEX.match(stripped) or (stripped.isupper() and 3 < len(stripped) < 60):
                    if stripped not in page_headings:
                        page_headings.append(stripped)

            extra: dict[str, object] = {
                "total_pages": total_pages,
            }
            if course_code:
                extra["course_code"] = course_code

            doc = Document.create(
                content=cleaned,
                source_file=path.name,
                file_type="pdf",
                category=doc_category,
                title=doc_title,
                page_number=page_num,
                headings=page_headings,
                extra=extra,
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} pages from PDF '{path.name}'.")
        return documents
