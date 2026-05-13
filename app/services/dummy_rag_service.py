from app.services.rag_interface import RAGServiceInterface
from app.models.schemas import QueryResponse

class DummyRAGService(RAGServiceInterface):
    
    def answer_question(self, query: str) -> QueryResponse:
        return QueryResponse(
            answer=f"This is a dummy answer to your question: '{query}'. The real AI is sleeping.",
            citations=["Dummy Document Page 1"]
        )