from app.services.rag_interface import RAGServiceInterface
from app.services.dummy_rag_service import DummyRAGService
from app.services.advanced_rag_service import AdvancedRAGService # *NEW*
from app.services.ingestion_service import IngestionService
from app.services.url_ingestion_service import UrlIngestionService
from app.services.github_ingestion_service import GithubIngestionService

rag_service = AdvancedRAGService() # <-- NEW! The system is now live.

ingestion_service = IngestionService() 
url_ingestion_service = UrlIngestionService(ingestion_service)

github_ingestion_service = GithubIngestionService(ingestion_service)

def get_github_ingestion_service() -> GithubIngestionService:
    return github_ingestion_service

def get_rag_service() -> RAGServiceInterface:
    return rag_service

def get_ingestion_service() -> IngestionService:
    return ingestion_service

def get_url_ingestion_service() -> UrlIngestionService:
    return url_ingestion_service