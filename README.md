# Project Overview

## What is this project about

---

This project is a **conversational AI assistant** designed to provide **helpful, accurate, and context-aware responses**.

The AI combines **local large language models (LLMs)** with **embedding-based search** to retrieve relevant knowledge and provide answers quickly.

---

## How this works

---

This project utilizes the following core technologies:

- **Ollama** : A lightweight local LLM runner that makes it easy to download, run, and manage large language models on your own machine without needing cloud APIs. It handles model execution, manages GPU/CPU resources, and serves models via a local API.

- **Mistral** : A fast, high-quality open-source LLM that excels in code understanding, reasoning, and multi-turn conversation. It powers the assistant’s ability to generate coherent, contextually relevant answers.

- **Nomic-Embed-Text** : A state-of-the-art text embedding model used to convert documents, code snippets, and user queries into vector representations. This allows the AI to **search** and **retrieve** the most relevant information from a knowledge base using similarity search.

Together, these components enable **Retrieval-Augmented Generation (RAG)**, meaning the assistant can search relevant context from stored data and feed it to the LLM for more accurate and grounded answers.

---

## Steps to follow to run this project

---

### **Step 1: Install dependencies**

Clone the repository and install the required dependencies.

```bash
# Clone repo
git clone https://github.com/prakashtaz0091/RAG-Chatbot.git
cd RAG-Chatbot


# Make virtual environment and activate
python3 -m venv venv
source venv/bin/activate # or in Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

```

---

### **Step 2: Install and run Ollama**

Download and install **Ollama** from:
[https://ollama.ai](https://ollama.ai)

Then, pull the required models:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

Run the Ollama server:

```bash
ollama serve
```

---

### **Step 3: Set up environment variables**

Create a `.env` file in the project root and add the necessary variables. Take a look at the `.env.example` file for reference.

Adjust variables according to your setup.

---

### **Step 4: Start the server**

Run the application:

```bash
python main.py
```

---

### **Step 5: Test the AI assistant**

You can interact with the AI assistant:

- Chat with it in the terminal

---

### Upcoming Improvements

- Expose API for external use
- Add a UI for better user experience
