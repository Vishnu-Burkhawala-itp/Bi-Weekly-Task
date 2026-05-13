from app.services.rag_interface import RAGServiceInterface
from app.services.dummy_rag_service import DummyRAGService
from app.services.advanced_rag_service import AdvancedRAGService # *NEW*
from app.services.ingestion_service import IngestionService

rag_service = AdvancedRAGService() # <-- NEW! The system is now live.

ingestion_service = IngestionService() 

def get_rag_service() -> RAGServiceInterface:
    return rag_service

def get_ingestion_service() -> IngestionService:
    return ingestion_service