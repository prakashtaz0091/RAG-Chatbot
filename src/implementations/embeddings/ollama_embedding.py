from langchain_community.embeddings import OllamaEmbeddings
from src.interfaces.embedding_interface import EmbeddingInterface


class OllamaEmbedding(EmbeddingInterface):
    """Ollama embedding implementation"""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.model_name, base_url=self.base_url)
