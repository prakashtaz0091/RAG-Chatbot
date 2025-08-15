from langchain_ollama import OllamaLLM
from src.interfaces.llm_interface import LLMInterface


class ImplementOllamaLLM(LLMInterface):
    """Ollama LLM implementation"""

    def __init__(
        self, model_name: str = "mistral", base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def get_llm(self):
        model = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.7,
            num_predict=512,
        )

        return model
