"""JSON document loader specifically optimized for structured student FAQs."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.rag.cleaner import clean_text
from app.rag.loaders.base import BaseLoader
from app.rag.schema import Document

logger = logging.getLogger(__name__)


class JSONLoader(BaseLoader):
    """Loads and transforms structured JSON datasets into standard Document objects."""

    supported_extensions: set[str] = {"json"}

    def load(self, file_path: Path, category: Optional[str] = None) -> list[Document]:
        """Loads a JSON file. Handles the LPU FAQ schema as well as generic JSON lists."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path.name}: {e}") from e

        documents: list[Document] = []
        doc_category = category or "faqs"

        # Schema 1: Standard LPU FAQ structure {"metadata": {...}, "faqs": [...]}
        if isinstance(data, dict) and "faqs" in data and isinstance(data["faqs"], list):
            meta_info = data.get("metadata", {})
            doc_title = meta_info.get("document_title", path.stem.replace("_", " "))

            for item in data["faqs"]:
                if not isinstance(item, dict):
                    continue
                q = clean_text(item.get("question", ""))
                a = clean_text(item.get("answer", ""))
                if not q and not a:
                    continue

                faq_id = item.get("faq_id", "")
                item_cat = item.get("category", doc_category)
                scope = item.get("applicable_scope", "")
                source = item.get("source", "")
                verif = item.get("verification_status", "")

                content_parts = [
                    f"Question: {q}",
                    f"Answer: {a}",
                ]
                if scope:
                    content_parts.append(f"Applicable Scope: {scope}")
                if item_cat:
                    content_parts.append(f"Category: {item_cat}")

                full_content = "\n".join(content_parts)

                extra_meta: dict[str, Any] = {
                    "faq_id": faq_id,
                    "verification_status": verif,
                    "source": source,
                    "applicable_scope": scope,
                }

                doc = Document.create(
                    content=full_content,
                    source_file=path.name,
                    file_type="json",
                    category=item_cat,
                    title=f"{doc_title} - {faq_id}" if faq_id else doc_title,
                    headings=[item_cat] if item_cat else [],
                    extra=extra_meta,
                    document_id=f"{path.stem}_{faq_id}" if faq_id else None,
                )
                documents.append(doc)

        # Schema 2: Array of FAQ items or documents
        elif isinstance(data, list):
            doc_title = path.stem.replace("_", " ")
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    q = clean_text(item.get("question", ""))
                    a = clean_text(item.get("answer", ""))
                    if q or a:
                        content = f"Question: {q}\nAnswer: {a}"
                    else:
                        content = clean_text(str(item))

                    item_cat = item.get("category", doc_category)
                    doc = Document.create(
                        content=content,
                        source_file=path.name,
                        file_type="json",
                        category=item_cat,
                        title=doc_title,
                        headings=[item_cat] if item_cat else [],
                        extra={"item_index": idx},
                    )
                    documents.append(doc)
                elif isinstance(item, str) and clean_text(item):
                    doc = Document.create(
                        content=clean_text(item),
                        source_file=path.name,
                        file_type="json",
                        category=doc_category,
                        title=doc_title,
                    )
                    documents.append(doc)

        # Schema 3: Generic dictionary
        elif isinstance(data, dict):
            doc_title = path.stem.replace("_", " ")
            content = clean_text(json.dumps(data, indent=2))
            if content:
                doc = Document.create(
                    content=content,
                    source_file=path.name,
                    file_type="json",
                    category=doc_category,
                    title=doc_title,
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} records from JSON '{path.name}'.")
        return documents
