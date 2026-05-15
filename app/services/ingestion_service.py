import os
from pathlib import Path
from typing import List

import chromadb
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from app.core.config import settings
from app.services.embedding_provider import get_embedding_model

class IngestionService:
    def __init__(self):
        # We tell the system where to look for PDFs and where to save the database
        self.data_dir = "./data"
        self.storage_dir = "./storage/chroma_db"

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up the API key as an environment variable so LlamaParse can find it
        os.environ["LLAMA_CLOUD_API_KEY"] = settings.LLAMA_CLOUD_API_KEY

    def run_pipeline(self) -> int:
        """Runs the entire multimodal ingestion pipeline and returns the number of leaves saved."""

        documents = self._load_pdf_documents()
        return self.index_documents(documents)

    def _load_pdf_documents(self) -> List[Document]:
        # --- STEP 1: The "Eyes" (Multimodal Parsing) ---
        # We configure LlamaParse to treat the PDFs as containing complex tables/diagrams.
        parser = LlamaParse(result_type="markdown", verbose=True)
        file_extractor = {".pdf": parser}

        # The robotic librarian reads the folder using our special parser eyes
        return SimpleDirectoryReader(
            input_dir=self.data_dir,
            file_extractor=file_extractor,
        ).load_data()

    def index_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        # --- STEP 2: The "Tree Builder" (Advanced Chunking) ---
        # We chop the document into Big Branches, Medium Twigs, and Tiny Leaves.
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        # We extract ONLY the tiny leaves to put into the search engine front-desk.
        leaf_nodes = get_leaf_nodes(nodes)

        # --- STEP 3: The "Translator" (Embeddings) ---
        # We download a fast, local translator to turn English into math vectors.
        embed_model = get_embedding_model()

        # --- STEP 4: The "Vault" (Vector Database) ---
        # We create a local database on your hard drive using ChromaDB
        db = chromadb.PersistentClient(path=self.storage_dir)
        chroma_collection = db.get_or_create_collection("tech_docs_collection")

        # We wrap the Chroma database in a LlamaIndex format
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Finally, we process the leaves through the translator and save them in the vault!
        VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        return len(leaf_nodes)