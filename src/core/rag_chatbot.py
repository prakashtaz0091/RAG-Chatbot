import os
from typing import Any, Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
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

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=self.config.memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the user's question.
            If you cannot find the answer in the context, say so clearly.
           
            Context: {context}
           
            Chat History: {chat_history}
            
            Human: {question}
            Assistant: """,
            input_variables=["context", "chat_history", "question"],
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
        """Setup the conversational retrieval chain"""
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Call setup_vectorstore() first."
            )

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.search_k},
        )

        # Create conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
        )

        logger.info("Conversational chain setup complete")

    def chat(self, question: str) -> Dict[str, Any]:
        """Process a chat message and return response with sources"""
        if not self.chain:
            raise ValueError("Chain not initialized. Call setup_chain() first.")

        try:
            response = self.chain({"question": question})

            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "chat_history": self.memory.chat_memory.messages,
            }
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "source_documents": [],
                "chat_history": self.memory.chat_memory.messages,
            }

    def clear_memory(self):
        """Clear chat memory"""
        self.memory.clear()
        logger.info("Chat memory cleared")

    def get_memory_summary(self) -> List[str]:
        """Get current chat history"""
        messages = self.memory.chat_memory.messages
        return [f"{msg.type}: {msg.content}" for msg in messages]
