from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    """Abstract interface for embedding models"""

    @abstractmethod
    def get_embeddings(self):
        """Return configured embedding instance"""
        pass
