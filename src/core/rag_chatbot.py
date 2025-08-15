import os
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.core.config import RAGConfig
from src.core.document_processor import DocumentProcessor
from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.llm_interface import LLMInterface
from src.interfaces.vectorstore_interface import VectorStoreInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChatbot:
    """Main RAG chatbot class with modular components"""

    def __init__(
        self,
        llm_interface: LLMInterface,
        embedding_interface: EmbeddingInterface,
        vectorstore_interface: VectorStoreInterface,
        config: RAGConfig = None,
    ):
        self.config = config or RAGConfig()
        self.llm_interface = llm_interface
        self.embedding_interface = embedding_interface
        self.vectorstore_interface = vectorstore_interface

        # Initialize components
        self.llm = llm_interface.get_llm()
        self.embeddings = embedding_interface.get_embeddings()
        self.vectorstore = None
        self.chain = None

        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the user's question.
            If you cannot find the answer in the context, say so clearly.
           
            Context: {context}
           
            Question: {question}
            Assistant: """,
            input_variables=["context", "question"],
        )

    def setup_vectorstore(
        self,
        documents_path: Optional[str] = None,
        vectorstore_path: Optional[str] = None,
    ):
        """Setup vector store from documents or load existing one"""
        documents_path = documents_path or self.config.documents_path
        vectorstore_path = vectorstore_path or self.config.vectorstore_path

        if vectorstore_path and os.path.exists(vectorstore_path):
            try:
                logger.info(f"Loading existing vector store from {vectorstore_path}")
                self.vectorstore = self.vectorstore_interface.load_vectorstore(
                    vectorstore_path, self.embeddings
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                logger.info("Creating new vector store...")
                self._create_new_vectorstore(documents_path, vectorstore_path)
        else:
            logger.info("Creating new vector store...")
            self._create_new_vectorstore(documents_path, vectorstore_path)

    def _create_new_vectorstore(
        self, documents_path: str, vectorstore_path: Optional[str]
    ):
        """Create new vector store from documents"""
        # Load and process documents
        doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )
        documents = doc_processor.load_markdown_documents(documents_path)

        if not documents:
            raise ValueError("No documents found to process")

        chunks = doc_processor.split_documents(documents)

        # Create vector store
        logger.info("Creating vector store...")
        self.vectorstore = self.vectorstore_interface.create_vectorstore(
            chunks, self.embeddings
        )

        # Save vector store if path provided
        if vectorstore_path:
            self.vectorstore_interface.save_vectorstore(
                self.vectorstore, vectorstore_path
            )
            logger.info(f"Vector store saved to {vectorstore_path}")

    def setup_chain(self):
        """Setup a retrieval QA chain (stateless)"""
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Call setup_vectorstore() first."
            )

        retriever = self.vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.search_k},
        )

        # No memory, pure retrieval
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template},
        )

    def chat(self, question: str) -> Dict[str, Any]:
        if not self.chain:
            raise ValueError("Chain not initialized. Call setup_chain() first.")

        try:
            response = self.chain.invoke({"query": question})
            return {
                "answer": response["result"],
                "source_documents": response.get("source_documents", []),
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": [],
            }
