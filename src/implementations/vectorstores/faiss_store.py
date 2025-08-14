from typing import List, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.interfaces.vectorstore_interface import VectorStoreInterface


class FAISSVectorStore(VectorStoreInterface):
    """FAISS vector store implementation"""

    def create_vectorstore(self, documents: List[Document], embeddings):
        return FAISS.from_documents(documents, embeddings)

    def load_vectorstore(self, path: str, embeddings):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    def save_vectorstore(self, vectorstore: Any, path: str) -> None:
        vectorstore.save_local(path)
