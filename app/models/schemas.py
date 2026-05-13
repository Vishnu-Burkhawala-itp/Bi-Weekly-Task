from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    citations: List[str]

class IngestionResponse(BaseModel):
    status: str
    nodes_processed: int