"""Abstract base class for document loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from app.rag.schema import Document


class BaseLoader(ABC):
    """Abstract base class for all file-specific document loaders."""

    supported_extensions: set[str] = set()

    def can_handle(self, file_path: Path) -> bool:
        """Checks if this loader supports the given file extension."""
        return file_path.suffix.lower().lstrip(".") in self.supported_extensions

    @abstractmethod
    def load(self, file_path: Path, category: Optional[str] = None) -> list[Document]:
        """Loads and parses the document into a list of standardized Document objects.

        Args:
            file_path: Path to the target file.
            category: Optional category override (e.g., syllabus, regulations).

        Returns:
            List of standardized Document objects.
        """
        pass
