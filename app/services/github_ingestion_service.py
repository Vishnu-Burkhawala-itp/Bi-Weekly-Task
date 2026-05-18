import base64
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

import requests
from llama_index.core.schema import Document

from app.core.config import settings
from app.services.ingestion_service import IngestionService


@dataclass
class GithubIngestionResult:
    nodes_processed: int
    processed_files: List[str]
    errors: List[str]


class GithubIngestionService:
    """
    Pulls GitHub repository contents using GitHub REST API
    and pushes documents into the SAME ingestion pipeline.
    """

    SUPPORTED_EXTENSIONS = {
        ".py",
        ".md",
        ".txt",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".yaml",
        ".yml",
        ".html",
        ".css",
        ".java",
        ".cpp",
        ".c",
        ".go",
        ".rs",
        ".sql",
    }

    EXCLUDED_FOLDERS = {
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        "__pycache__",
        "venv",
    }

    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service

        self.headers = {
            "Accept": "application/vnd.github+json"
        }

        if settings.GITHUB_TOKEN:
            self.headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    def run_from_github_repo(self, repo_url: str) -> GithubIngestionResult:
        owner, repo = self._extract_repo_details(repo_url)

        documents: List[Document] = []
        processed_files: List[str] = []
        errors: List[str] = []

        self._fetch_repo_recursive(
            owner=owner,
            repo=repo,
            path="",
            documents=documents,
            processed_files=processed_files,
            errors=errors,
        )

        if not documents:
            raise ValueError("No supported files found in repository.")

        nodes_processed = self.ingestion_service.index_documents(documents)

        return GithubIngestionResult(
            nodes_processed=nodes_processed,
            processed_files=processed_files,
            errors=errors,
        )

    def _extract_repo_details(self, repo_url: str):
        parsed = urlparse(repo_url)

        if "github.com" not in parsed.netloc:
            raise ValueError("Only GitHub repositories are supported.")

        parts = parsed.path.strip("/").split("/")

        if len(parts) < 2:
            raise ValueError("Invalid GitHub repository URL.")

        return parts[0], parts[1]

    def _fetch_repo_recursive(
        self,
        owner,
        repo,
        path,
        documents,
        processed_files,
        errors,
    ):
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

        response = requests.get(api_url, headers=self.headers, timeout=30)

        if response.status_code != 200:
            raise ValueError(f"GitHub API error: {response.text}")

        items = response.json()

        if isinstance(items, dict):
            items = [items]

        for item in items:
            try:
                item_type = item.get("type")
                item_path = item.get("path")

                if not item_path:
                    continue

                if item_type == "dir":
                    folder_name = item_path.split("/")[-1]

                    if folder_name in self.EXCLUDED_FOLDERS:
                        continue

                    self._fetch_repo_recursive(
                        owner,
                        repo,
                        item_path,
                        documents,
                        processed_files,
                        errors,
                    )

                elif item_type == "file":
                    if not self._is_supported_file(item_path):
                        continue

                    file_response = requests.get(
                        item["url"],
                        headers=self.headers,
                        timeout=30,
                    )

                    if file_response.status_code != 200:
                        continue

                    file_data = file_response.json()

                    content = file_data.get("content", "")

                    if not content:
                        continue

                    decoded = base64.b64decode(content).decode(
                        "utf-8",
                        errors="ignore"
                    )

                    metadata = {
                        "file_name": item_path,
                        "github_repo": f"{owner}/{repo}",
                        "source": "github",
                    }

                    documents.append(
                        Document(
                            text=decoded,
                            metadata=metadata,
                        )
                    )

                    processed_files.append(item_path)

            except Exception as exc:
                errors.append(f"{item.get('path', 'unknown')} -> {exc}")

    def _is_supported_file(self, file_path: str) -> bool:
        return any(
            file_path.lower().endswith(ext)
            for ext in self.SUPPORTED_EXTENSIONS
        )