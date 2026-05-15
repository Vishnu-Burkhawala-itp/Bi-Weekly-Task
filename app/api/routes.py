from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionResponse,
    UrlIngestionRequest,
    UrlIngestionResponse,
)
from app.services.rag_interface import RAGServiceInterface
from app.services.ingestion_service import IngestionService
from app.services.url_ingestion_service import UrlIngestionService
from app.api.dependencies import (
    get_rag_service,
    get_ingestion_service,
    get_url_ingestion_service,
)

router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
def ask_question(
    request: QueryRequest, 
    service: RAGServiceInterface = Depends(get_rag_service) 
):
    return service.answer_question(request.query)

# *NEW* The trigger button for Phase 2
@router.post("/ingest", response_model=IngestionResponse)
def ingest_documents(
    service: IngestionService = Depends(get_ingestion_service)
):
    try:
        # Run the massive pipeline
        leaves_count = service.run_pipeline()
        return IngestionResponse(
            status="Success! PDFs were seen, chopped into a tree, translated to math, and locked in the vault.",
            nodes_processed=leaves_count
        )
    except Exception as e:
        # If the API key is missing or the data folder is empty, gracefully fail.
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/urls", response_model=UrlIngestionResponse)
def ingest_urls(
    request: UrlIngestionRequest,
    service: UrlIngestionService = Depends(get_url_ingestion_service),
):
    try:
        result = service.run_from_urls(request.urls)
        return UrlIngestionResponse(
            status="Success! URLs were fetched and indexed.",
            nodes_processed=result.nodes_processed,
            urls_processed=result.processed_urls,
            errors=result.errors,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))