from abc import ABC, abstractmethod
from typing import List, Any
from langchain.schema import Document


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores"""

    @abstractmethod
    def create_vectorstore(self, documents: List[Document], embeddings) -> Any:
        """Create vector store from documents"""
        pass

    @abstractmethod
    def load_vectorstore(self, path: str, embeddings) -> Any:
        """Load existing vector store"""
        pass

    @abstractmethod
    def save_vectorstore(self, vectorstore: Any, path: str) -> None:
        """Save vector store to path"""
        pass
