from fastapi import FastAPI
from main import init_chatbot
from pydantic import BaseModel


app = FastAPI(title="RAG Chatbot API")


chatbot = init_chatbot()
if chatbot is None:
    raise Exception("Failed to initialize chatbot")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    result = chatbot.chat(request.message)
    return {
        "answer": result["answer"],
        "sources": [
            doc.metadata.get("source", "Unknown") for doc in result["source_documents"]
        ],
    }
