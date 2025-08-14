from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract interface for LLM models"""
    
    @abstractmethod
    def get_llm(self):
        """Return configured LLM instance"""
        pass