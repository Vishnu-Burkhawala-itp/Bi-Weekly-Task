from abc import ABC, abstractmethod
from app.models.schemas import QueryResponse

class RAGServiceInterface(ABC):
    
    @abstractmethod
    def answer_question(self, query: str) -> QueryResponse:
        """
        Every RAG service we ever build MUST have this exact method.
        It takes a string, and returns a QueryResponse. No exceptions.
        """
        pass