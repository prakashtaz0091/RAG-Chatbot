import os
from src.core.config import RAGConfig
from src.core.rag_chatbot import RAGChatbot
from src.implementations.llms.ollama_llm import OllamaLLM
from src.implementations.embeddings.ollama_embedding import OllamaEmbedding
from src.implementations.vectorstores.faiss_store import FAISSVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_chatbot():
    """Initialize the RAG Chatbot for API usage"""
    # Load configuration
    config = RAGConfig.from_env()

    # Initialize components with dependency injection
    llm_interface = OllamaLLM(
        model_name=config.ollama_llm_model, base_url=config.ollama_base_url
    )
    embedding_interface = OllamaEmbedding(
        model_name=config.ollama_embedding_model, base_url=config.ollama_base_url
    )
    vectorstore_interface = FAISSVectorStore()

    # Create chatbot
    chatbot = RAGChatbot(
        llm_interface=llm_interface,
        embedding_interface=embedding_interface,
        vectorstore_interface=vectorstore_interface,
        config=config,
    )

    try:
        chatbot.setup_vectorstore()  # Setup vector store
        chatbot.setup_chain()  # Setup chain
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        return None
    else:
        print("🤖 RAG Chatbot initialized successfully!")
        return chatbot


def main():
    """Main function to run the RAG chatbot"""

    chatbot = init_chatbot()
    if chatbot is None:
        return

    try:
        print("Type 'quit' to exit, 'clear' to clear memory\n")

        # Interactive chat loop
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                chatbot.clear_memory()
                print("✨ Chat memory cleared!")
                continue
            elif not user_input:
                continue

            # Get response
            result = chatbot.chat(user_input)
            print(f"\nBot: {result['answer']}")

            # Show sources (optional)
            if result["source_documents"]:
                print("\n📚 Sources:")
                for i, doc in enumerate(result["source_documents"][:2]):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  {i + 1}. {os.path.basename(source)}")

            print("-" * 50)

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
