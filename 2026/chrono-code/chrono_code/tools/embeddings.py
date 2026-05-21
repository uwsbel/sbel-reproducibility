"""Local sentence-transformers wrapper for semantic retrieval.

Singleton-loaded (first call pays the ~1-2 s model load; subsequent calls
are free). All embeddings are L2-normalized so downstream cosine becomes
a plain dot product.

Used by:
- `scripts/build_api_index.py` to produce `api_embeddings.npy` at build time
- `chrono_code.tools.api_rag.VectorIndex` to embed the user's question at
  query time

The model is chosen via `settings.embedding_model` (default
`all-MiniLM-L6-v2`, 384-dim). Runs fully local once the weights are
cached in `~/.cache/huggingface/hub/`.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_name: Optional[str] = None
_lock = threading.Lock()


def _load_model(model_name: str):
    """Lazy, thread-safe singleton loader for the sentence-transformers model."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    with _lock:
        if _model is not None and _model_name == model_name:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it via "
                "`uv sync` (it's declared in pyproject.toml)."
            ) from exc
        logger.info("[embeddings] loading model %s", model_name)
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info(
            "[embeddings] model loaded: %s (dim=%d)",
            model_name, _model.get_sentence_embedding_dimension(),
        )
    return _model


def get_embedding_dim(model_name: str = "all-MiniLM-L6-v2") -> int:
    return int(_load_model(model_name).get_sentence_embedding_dimension())


def encode(
    texts: Iterable[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode a batch of strings into an L2-normalized float32 matrix.

    Returns shape `(N, D)`. Safe to cast to float16 for on-disk storage;
    the cosine score degrades by <1e-3 versus float32 in practice.
    """
    model = _load_model(model_name)
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype(np.float32, copy=False)


def encode_query(
    text: str,
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode a single query string. Returns shape `(D,)`, L2-normalized."""
    vec = encode([text], model_name=model_name)[0]
    return vec
