"""Document ingestion service for batch processing and format dispatch."""

import json
import logging
from pathlib import Path
from typing import Optional

from app.rag.loaders.base import BaseLoader
from app.rag.loaders.docx_loader import DOCXLoader
from app.rag.loaders.json_loader import JSONLoader
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.schema import Document, IngestionReport

logger = logging.getLogger(__name__)


class UnsupportedFileError(Exception):
    """Raised when an unsupported file format is encountered."""
    pass


class DocumentIngestionService:
    """Manages document extraction across different formats with robust error handling."""

    def __init__(self, loaders: Optional[list[BaseLoader]] = None):
        self.loaders = loaders or [
            PDFLoader(),
            DOCXLoader(),
            JSONLoader(),
        ]

    def _get_loader_for_path(self, path: Path) -> Optional[BaseLoader]:
        """Finds the loader matching the file extension."""
        for loader in self.loaders:
            if loader.can_handle(path):
                return loader
        return None

    def infer_category_from_path(self, path: Path) -> str:
        """Infers the document category from directory names or filename."""
        parent_names = [p.name.lower() for p in path.parents]
        stem_lower = path.stem.lower()

        if "syllabus" in parent_names or "syllabus" in stem_lower:
            return "syllabus"
        if "regulations" in parent_names or "regulation" in stem_lower:
            return "regulations"
        if "notices" in parent_names or "notice" in stem_lower or "exam" in stem_lower:
            return "notices"
        if "faqs" in parent_names or "faq" in stem_lower:
            return "faqs"
        if "policies" in parent_names or "policy" in stem_lower:
            return "policies"

        return "general"

    def ingest_file(self, file_path: Path, category: Optional[str] = None) -> list[Document]:
        """Ingests a single file using the appropriate loader."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        loader = self._get_loader_for_path(path)
        if not loader:
            raise UnsupportedFileError(f"Unsupported file format: {path.suffix}")

        cat = category or self.infer_category_from_path(path)
        return loader.load(path, category=cat)

    def ingest_directory(
        self,
        directory_path: Path,
        recursive: bool = True,
    ) -> tuple[list[Document], IngestionReport]:
        """Scans and extracts all supported documents in a directory.

        Guarantees that individual extraction failures do not abort the entire batch.
        Uses JSON as the authoritative source of truth for FAQs if dual DOCX/JSON exists.
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pattern = "**/*" if recursive else "*"
        all_paths = [p for p in dir_path.glob(pattern) if p.is_file()]

        # Identify if JSON FAQs exist so redundant DOCX FAQs are skipped
        has_faq_json = any(
            p.suffix.lower() == ".json" and "faq" in p.stem.lower()
            for p in all_paths
        )

        all_documents: list[Document] = []
        report = IngestionReport()

        for path in all_paths:
            report.total_files_scanned += 1

            # Rule: Use JSON as the FAQ ingestion source of truth
            if has_faq_json and path.suffix.lower() == ".docx" and "faq" in path.stem.lower():
                report.skipped_files.append(
                    f"{path.name} (Skipped redundant FAQ docx; JSON is canonical source)"
                )
                logger.info(f"Skipping redundant FAQ docx '{path.name}', using JSON dataset.")
                continue

            loader = self._get_loader_for_path(path)
            if not loader:
                report.unsupported_files.append(path.name)
                logger.debug(f"Skipping unsupported file format: {path.name}")
                continue

            try:
                cat = self.infer_category_from_path(path)
                docs = loader.load(path, category=cat)
                all_documents.extend(docs)
                report.successful_files.append(path.name)
                report.total_documents_extracted += len(docs)
                report.documents_by_category[cat] = (
                    report.documents_by_category.get(cat, 0) + len(docs)
                )
            except Exception as e:
                logger.error(f"Failed to ingest '{path.name}': {e}", exc_info=True)
                report.failed_files.append({
                    "file": path.name,
                    "error": str(e),
                })

        return all_documents, report

    def save_documents(self, documents: list[Document], output_path: Path) -> None:
        """Serializes extracted documents to a JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        serialized = [doc.model_dump() for doc in documents]
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents to {out_p}")
