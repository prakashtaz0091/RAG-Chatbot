from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_markdown_documents(self, directory_path: str) -> List[Document]:
        """Load all markdown files from directory"""
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True,
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} markdown documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
