import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.dependencies import (
    get_ingestion_service,
    get_rag_service,
    get_url_ingestion_service,
    get_github_ingestion_service
)
from app.services.ingestion_service import IngestionService
from app.services.rag_interface import RAGServiceInterface
from app.services.url_ingestion_service import UrlIngestionService
from app.services.github_ingestion_service import GithubIngestionService

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter()


def _render_home(
    request: Request,
    query: str = "",
    answer: str = "",
    citations: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None,
    ingest_status: str = "",
    ingest_nodes: Optional[int] = None,
    url_input: str = "",
    url_status: str = "",
    url_nodes: Optional[int] = None,
    url_processed: Optional[List[str]] = None,
    url_errors: Optional[List[str]] = None,
    data_dir: str = "",
    error: str = "",
    github_input: str = "",
    github_status: str = "",
    github_nodes: Optional[int] = None,
    github_processed: Optional[List[str]] = None,
    github_errors: Optional[List[str]] = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
            "citations": citations or [],
            "uploaded_files": uploaded_files or [],
            "ingest_status": ingest_status,
            "ingest_nodes": ingest_nodes,
            "url_input": url_input,
            "url_status": url_status,
            "url_nodes": url_nodes,
            "url_processed": url_processed or [],
            "url_errors": url_errors or [],
            "data_dir": data_dir,
            "error": error,
            "github_input": github_input,
            "github_status": github_status,
            "github_nodes": github_nodes,
            "github_processed": github_processed or [],
            "github_errors": github_errors or [],
        },
    )


@router.get("/ui", response_class=HTMLResponse)
def ui_home(
    request: Request,
    service: IngestionService = Depends(get_ingestion_service),
) -> HTMLResponse:
    return _render_home(request, data_dir=service.data_dir)


@router.post("/ui/ask", response_class=HTMLResponse)
def ui_ask(
    request: Request,
    query: str = Form(...),
    service: RAGServiceInterface = Depends(get_rag_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> HTMLResponse:
    try:
        response = service.answer_question(query)
        return _render_home(
            request,
            query=query,
            answer=response.answer,
            citations=response.citations,
            data_dir=ingestion_service.data_dir,
        )
    except Exception as exc:
        return _render_home(
            request,
            query=query,
            data_dir=ingestion_service.data_dir,
            error=str(exc),
        )


@router.post("/ui/ingest", response_class=HTMLResponse)
def ui_ingest(
    request: Request,
    files: Optional[List[UploadFile]] = File(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> HTMLResponse:
    try:
        saved_files: List[str] = []
        data_dir = Path(service.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if files:
            for upload in files:
                if not upload.filename:
                    continue
                filename = Path(upload.filename).name
                if not filename.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are supported.")
                target_path = data_dir / filename
                with target_path.open("wb") as buffer:
                    shutil.copyfileobj(upload.file, buffer)
                upload.file.close()
                saved_files.append(filename)

        nodes_processed = service.run_pipeline()
        status = "Ingestion complete. Documents indexed."
        return _render_home(
            request,
            uploaded_files=saved_files,
            ingest_status=status,
            ingest_nodes=nodes_processed,
            data_dir=service.data_dir,
        )
    except Exception as exc:
        return _render_home(request, data_dir=service.data_dir, error=str(exc))


@router.post("/ui/ingest/urls", response_class=HTMLResponse)
def ui_ingest_urls(
    request: Request,
    urls: str = Form(...),
    service: UrlIngestionService = Depends(get_url_ingestion_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> HTMLResponse:
    url_list = [line.strip() for line in urls.splitlines() if line.strip()]
    try:
        result = service.run_from_urls(url_list)
        status = "URL ingestion complete. Pages indexed."
        return _render_home(
            request,
            url_input=urls,
            url_status=status,
            url_nodes=result.nodes_processed,
            url_processed=result.processed_urls,
            url_errors=result.errors,
            data_dir=ingestion_service.data_dir,
        )
    except Exception as exc:
        return _render_home(
            request,
            url_input=urls,
            data_dir=ingestion_service.data_dir,
            error=str(exc),
        )


@router.post("/ui/ingest/github", response_class=HTMLResponse)
def ui_ingest_github(
    request: Request,
    repo_url: str = Form(...),
    service: GithubIngestionService = Depends(get_github_ingestion_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> HTMLResponse:

    try:
        result = service.run_from_github_repo(repo_url)

        return _render_home(
            request,
            github_input=repo_url,
            github_status="GitHub repository indexed successfully.",
            github_nodes=result.nodes_processed,
            github_processed=result.processed_files,
            github_errors=result.errors,
            data_dir=ingestion_service.data_dir,
        )

    except Exception as exc:
        return _render_home(
            request,
            github_input=repo_url,
            data_dir=ingestion_service.data_dir,
            error=str(exc),
        )
