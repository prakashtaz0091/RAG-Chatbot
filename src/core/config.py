import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration class for RAG chatbot"""

    # Paths
    documents_path: str = "./data/documents"
    vectorstore_path: str = "./data/vectorstore"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Memory
    memory_window: int = 5

    # Retrieval
    search_k: int = 4
    search_type: str = "similarity"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            documents_path=os.getenv("RAG_DOCUMENTS_PATH", cls.documents_path),
            vectorstore_path=os.getenv("RAG_VECTORSTORE_PATH", cls.vectorstore_path),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            memory_window=int(os.getenv("RAG_MEMORY_WINDOW", cls.memory_window)),
            search_k=int(os.getenv("RAG_SEARCH_K", cls.search_k)),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_llm_model=os.getenv("OLLAMA_LLM_MODEL", cls.ollama_llm_model),
            ollama_embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL", cls.ollama_embedding_model
            ),
        )
