from langchain_openai import ChatOpenAI
from src.interfaces.llm_interface import LLMInterface


class ImplementOpenAILLM(LLMInterface):
    """OpenAI LLM implementation"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return ChatOpenAI(model=self.model_name, temperature=self.temperature)
