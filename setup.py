from setuptools import setup, find_packages

setup(
    name="rag-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-openai>=0.0.8",
        "faiss-cpu>=1.7.4",
        "unstructured>=0.11.0",
        "markdown>=3.5.0",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    description="Modular RAG Chatbot with LangChain",
    python_requires=">=3.8",
)