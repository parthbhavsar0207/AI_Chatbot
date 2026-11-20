"""Document loaders for PDF, DOCX, and JSON academic sources."""

from app.rag.loaders.base import BaseLoader
from app.rag.loaders.docx_loader import DOCXLoader
from app.rag.loaders.json_loader import JSONLoader
from app.rag.loaders.pdf_loader import PDFLoader

__all__ = ["BaseLoader", "PDFLoader", "DOCXLoader", "JSONLoader"]
