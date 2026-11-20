"""DOCX document loader preserving heading hierarchy and table structures."""

import logging
from pathlib import Path
import re
from typing import Optional
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.rag.cleaner import clean_text
from app.rag.loaders.base import BaseLoader
from app.rag.schema import Document

logger = logging.getLogger(__name__)


class DOCXLoader(BaseLoader):
    """Extracts text from DOCX documents preserving headings, paragraphs, and tables."""

    supported_extensions: set[str] = {"docx"}

    @staticmethod
    def _get_heading_level(style: str, content: str) -> Optional[int]:
        """Detects if paragraph is a heading and determines its numeric hierarchy level."""
        style_lower = style.lower().strip()
        if "title" in style_lower:
            return 0
        if "subtitle" in style_lower:
            return 1
        if "heading 1" in style_lower:
            return 1
        if "heading 2" in style_lower:
            return 2
        if "heading 3" in style_lower:
            return 3
        if "heading 4" in style_lower:
            return 4
        if "heading" in style_lower:
            digits = re.findall(r"\d+", style_lower)
            return int(digits[0]) if digits else 1
        # Heuristic for short uppercase section headers
        if content.isupper() and 3 < len(content) < 80:
            return 1
        return None

    def _format_table(self, table: Table) -> str:
        """Serializes a docx table into clean Markdown table format."""
        rows_data: list[list[str]] = []
        for row in table.rows:
            row_cells = [clean_text(cell.text).replace("\n", " ") for cell in row.cells]
            # Deduplicate consecutive identical merged cells in the same row
            cleaned_row: list[str] = []
            for cell in row_cells:
                if not cleaned_row or cell != cleaned_row[-1]:
                    cleaned_row.append(cell)
            if any(cleaned_row):
                rows_data.append(cleaned_row)

        if not rows_data:
            return ""

        # Normalize column count
        max_cols = max(len(r) for r in rows_data)
        padded_rows = [r + [""] * (max_cols - len(r)) for r in rows_data]

        lines: list[str] = []
        header = padded_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in padded_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def load(self, file_path: Path, category: Optional[str] = None) -> list[Document]:
        """Loads a DOCX document, extracting sections based on headings."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        doc = docx.Document(str(path))
        doc_category = category or ("notices" if "notice" in path.name.lower() else "regulations")
        doc_title = path.stem.replace("_", " ").strip()

        # Extract blocks in sequential order
        blocks: list[tuple[str, str, str]] = []  # (block_type, text_or_table, style_name)

        # Check document title from first non-empty paragraph or core properties
        if doc.core_properties and doc.core_properties.title:
            extracted_title = doc.core_properties.title.strip()
            if extracted_title:
                doc_title = extracted_title

        for child in doc.element.body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                p = Paragraph(child, doc)
                text = p.text.strip()
                if text:
                    style_name = p.style.name if p.style else ""
                    blocks.append(("paragraph", text, style_name))
            elif tag == "tbl":
                tbl = Table(child, doc)
                tbl_md = self._format_table(tbl)
                if tbl_md:
                    blocks.append(("table", tbl_md, "Table"))

        if not blocks:
            logger.warning(f"DOCX file has no extractable content: {path.name}")
            return []

        # Check if first paragraph represents document title
        if blocks and blocks[0][0] == "paragraph":
            first_style = blocks[0][2].lower()
            if "title" in first_style or "heading 1" in first_style:
                doc_title = blocks[0][1]

        # Group blocks by major headings to preserve logical document units
        sections: list[dict[str, object]] = []
        current_heading = doc_title
        current_content_parts: list[str] = []
        heading_stack: list[tuple[int, str]] = [(0, doc_title)]

        for block_type, content, style in blocks:
            heading_level = self._get_heading_level(style, content) if block_type == "paragraph" else None
            is_heading = heading_level is not None

            if is_heading:
                if current_content_parts:
                    section_text = clean_text("\n\n".join(current_content_parts))
                    if section_text:
                        full_content = (
                            f"{current_heading}\n\n{section_text}"
                            if current_heading and current_heading != doc_title and not section_text.startswith(current_heading)
                            else section_text
                        )
                        sections.append({
                            "heading": current_heading,
                            "headings": [h for _, h in heading_stack],
                            "content": full_content,
                        })
                    current_content_parts = []

                # Pop sibling and deeper heading levels
                while heading_stack and heading_stack[-1][0] >= heading_level:
                    heading_stack.pop()
                heading_stack.append((heading_level, content))
                current_heading = content
            else:
                current_content_parts.append(content)

        # Flush final section
        if current_content_parts:
            section_text = clean_text("\n\n".join(current_content_parts))
            if section_text:
                full_content = (
                    f"{current_heading}\n\n{section_text}"
                    if current_heading and current_heading != doc_title and not section_text.startswith(current_heading)
                    else section_text
                )
                sections.append({
                    "heading": current_heading,
                    "headings": [h for _, h in heading_stack],
                    "content": full_content,
                })

        # If no sections were separated (e.g. short notice or continuous text), create single document
        if not sections:
            all_content = clean_text("\n\n".join(b[1] for b in blocks))
            if all_content:
                sections.append({
                    "heading": doc_title,
                    "headings": [doc_title],
                    "content": all_content,
                })

        documents: list[Document] = []
        for sec in sections:
            doc_obj = Document.create(
                content=str(sec["content"]),
                source_file=path.name,
                file_type="docx",
                category=doc_category,
                title=doc_title,
                headings=sec["headings"] if isinstance(sec["headings"], list) else [],
                extra={"section": sec["heading"]},
            )
            documents.append(doc_obj)

        logger.info(f"Loaded {len(documents)} sections from DOCX '{path.name}'.")
        return documents
