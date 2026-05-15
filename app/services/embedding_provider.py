from pathlib import Path
from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CACHE_DIR = Path("./storage/hf_cache")
_embed_model: Optional[HuggingFaceEmbedding] = None


def _ensure_cache_dir() -> str:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(_CACHE_DIR)


def get_embedding_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        cache_dir = _ensure_cache_dir()
        _embed_model = HuggingFaceEmbedding(
            model_name=_MODEL_NAME,
            cache_folder=cache_dir,
        )
    return _embed_model
