from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    citations: List[str]

class IngestionResponse(BaseModel):
    status: str
    nodes_processed: int


class UrlIngestionRequest(BaseModel):
    urls: List[str]


class UrlIngestionResponse(BaseModel):
    status: str
    nodes_processed: int
    urls_processed: List[str]
    errors: List[str] = Field(default_factory=list)