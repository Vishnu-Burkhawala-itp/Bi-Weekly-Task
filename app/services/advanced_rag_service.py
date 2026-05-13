import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings 
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI # We still use the OpenAI class!
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from app.services.rag_interface import RAGServiceInterface
from app.models.schemas import QueryResponse
from app.core.config import settings

class AdvancedRAGService(RAGServiceInterface):
    def __init__(self):
        # Pass the GitHub PAT (stored in OPENAI_API_KEY) to the environment
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        
        # --- 1. The Translator ---
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # --- 2. The Writer (Updated for GitHub Models) ---
        # We explicitly tell LlamaIndex to send the request to Azure/GitHub 
        # instead of the default OpenAI servers.
        Settings.llm = OpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            api_base="https://models.inference.ai.azure.com", # The GitHub Models endpoint
            api_key=settings.OPENAI_API_KEY # Your github_pat_...
        )
        
        # --- Connect to Vault ---
        db = chromadb.PersistentClient(path="./storage/chroma_db")
        chroma_collection = db.get_or_create_collection("tech_docs_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load the index
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        
        # --- Context Reassembly ---
        self.postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        
        # --- Stage 2: Cohere Rerank ---
        self.reranker = CohereRerank(
            api_key=settings.COHERE_API_KEY, 
            top_n=5, 
            model="rerank-v4.0-pro"
        )

        # --- Build Query Engine ---
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=30,
            node_postprocessors=[self.postprocessor, self.reranker],
        )

    def answer_question(self, query: str) -> QueryResponse:
        response = self.query_engine.query(query)
        
        citations = []
        for node in response.source_nodes:
            file_name = node.metadata.get('file_name', 'Unknown Document')
            text_snippet = node.text[:150] + "..." 
            citations.append(f"Source: {file_name} | Snippet: {text_snippet}")
            
        return QueryResponse(
            answer=str(response),
            citations=citations
        )