"""PyChrono API RAG: hybrid retrieval + Haiku-backed Q&A.

Two indexes, built by ``scripts/build_api_index.py``:
  * ``api_chunks.json``        — flat list, one chunk per class/function/const
  * ``api_embeddings.npy``     — L2-normalized float16 matrix (shape [N, D])
  * ``api_embeddings.meta.json`` — model/dim/count (for staleness check)

Retrieval fuses two rankers:
  1. **Keyword**: CamelCase-aware substring scoring over chunk name + keywords.
     Wins on exact API names (``ChBodyEasyBox``, ``SetTireType``).
  2. **Semantic**: cosine against the L2-normalized query vector. Wins on
     natural-language queries (``"how do I attach a sensor to a body?"``).

Fusion uses Reciprocal Rank Fusion (RRF, k=60) — parameter-free and
robust to score-scale mismatch between the two rankers. Falls back to
keyword-only if the .npy is missing or stale.

Usage (standalone):
    from chrono_agent.tools.api_rag import ApiQueryAgent
    agent = ApiQueryAgent()
    answer = await agent.query("How do I set mass and position on ChBody?")
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

import anthropic
import numpy as np

from chrono_agent.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "pychrono_docs" / "api_chunks.json"
)
_DEFAULT_VECTOR_PATH = _DEFAULT_INDEX_PATH.with_name("api_embeddings.npy")
_DEFAULT_VECTOR_META_PATH = _DEFAULT_INDEX_PATH.with_name("api_embeddings.meta.json")

# RRF constant — the paper's default. Small enough that the top ranks
# dominate but large enough to keep a long tail in play.
_RRF_K = 60

_SYSTEM_PROMPT = (
    "You answer questions about the PyChrono physics simulation library. "
    "Use ONLY the API documentation chunks provided in the user message — "
    "do not invent signatures, parameters, or behaviours not present in those chunks. "
    "If the answer is not in the provided chunks, say so explicitly. "
    "Keep answers tight (≤ 300 words). When asked about a method or constructor, "
    "quote its exact signature verbatim from the chunk."
)


class VectorIndex:
    """L2-normalized embedding matrix for semantic retrieval.

    Holds a float16 ``(N, D)`` matrix plus the metadata file that records
    the embedding model. At query time the question is embedded with the
    same model and ranked by dot product (= cosine, since everything is
    L2-normalized at build + query time).

    Staleness check: if the .npy row count doesn't match the JSON chunk
    count, or the meta file is missing, the index is treated as disabled.
    """

    def __init__(
        self,
        *,
        vector_path: Path,
        meta_path: Path,
        expected_count: int,
    ) -> None:
        self.available: bool = False
        self.matrix: Optional[np.ndarray] = None
        self.model_name: Optional[str] = None
        self.dim: int = 0

        if not vector_path.exists() or not meta_path.exists():
            logger.info(
                "[VectorIndex] disabled — %s or %s missing. Run "
                "`python scripts/build_api_index.py` to enable semantic retrieval.",
                vector_path.name, meta_path.name,
            )
            return

        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            matrix = np.load(vector_path).astype(np.float32, copy=False)
        except Exception as exc:
            logger.warning("[VectorIndex] load failed: %s — disabling vector retrieval", exc)
            return

        count = int(meta.get("count", -1))
        if count != expected_count or matrix.shape[0] != expected_count:
            logger.warning(
                "[VectorIndex] stale — meta.count=%d matrix.rows=%d chunks=%d. "
                "Re-run scripts/build_api_index.py. Disabling vector retrieval.",
                count, matrix.shape[0], expected_count,
            )
            return

        self.matrix = matrix
        self.model_name = str(meta.get("model") or "all-MiniLM-L6-v2")
        self.dim = int(meta.get("dim") or matrix.shape[1])
        self.available = True
        logger.info(
            "[VectorIndex] loaded %d vectors dim=%d model=%s",
            matrix.shape[0], self.dim, self.model_name,
        )

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_index, cosine_score)`` sorted descending.

        Empty result if the index is unavailable, the query is blank, or
        the embedding model can't be loaded at query time.
        """
        if not self.available or self.matrix is None or not query.strip():
            return []
        try:
            from chrono_agent.tools.embeddings import encode_query
            qvec = encode_query(query, model_name=self.model_name or "all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("[VectorIndex] query embedding failed: %s", exc)
            return []

        scores = self.matrix @ qvec.astype(np.float32, copy=False)
        k = max(1, min(top_k, scores.shape[0]))
        # argpartition is O(N); followed by a small sort on the k survivors.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]


def _rrf_fuse(
    ranked_lists: list[list[int]],
    *,
    k: int = _RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked index lists.

    Score for chunk ``i`` is ``sum_j 1 / (k + rank_j(i))`` over all rankers
    that placed ``i``. Parameter-free (no score-scale matching needed) and
    known to beat either ranker alone on most hybrid-retrieval benchmarks.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class ApiChunkIndex:
    """Loads api_chunks.json (+ optional api_embeddings.npy) and provides
    keyword / vector / hybrid retrieval.

    Thread-safe singleton load (lazy, first use).
    """

    _instance: Optional["ApiChunkIndex"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        path: Path = _DEFAULT_INDEX_PATH,
        *,
        vector_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> None:
        self._chunks: list[dict[str, Any]] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._chunks = json.load(f)
            logger.info("[ApiChunkIndex] loaded %d chunks from %s", len(self._chunks), path)
        else:
            logger.warning(
                "[ApiChunkIndex] index not found at %s — run scripts/build_api_index.py", path
            )

        self._vector = VectorIndex(
            vector_path=vector_path or path.with_name("api_embeddings.npy"),
            meta_path=meta_path or path.with_name("api_embeddings.meta.json"),
            expected_count=len(self._chunks),
        )

    @classmethod
    def get(cls, path: Path = _DEFAULT_INDEX_PATH) -> "ApiChunkIndex":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(path)
        return cls._instance

    @property
    def vector_available(self) -> bool:
        return self._vector.available

    # ------------------------------------------------------------------
    # Rankers — each returns a list of chunk indices (not chunks), so they
    # can be fused by RRF. Convert to chunks only at the end.
    # ------------------------------------------------------------------

    def _keyword_rank(self, query: str, top_k: int) -> list[int]:
        """Keyword scoring, CamelCase-aware. Returns ranked chunk indices."""
        if not self._chunks:
            return []

        raw_tokens = re.findall(r"[a-zA-Z]{2,}", query)
        query_tokens = [t.lower() for t in raw_tokens]
        expanded: list[str] = []
        for t in raw_tokens:
            parts = re.sub(r"([A-Z][a-z]+)", r" \1", t).split()
            expanded.extend(p.lower() for p in parts if len(p) >= 2)
        query_tokens = list(set(query_tokens + expanded))

        if not query_tokens:
            return list(range(min(top_k, len(self._chunks))))

        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self._chunks):
            score = 0.0
            name_lower = chunk["name"].lower()
            kw_set: set[str] = set(chunk.get("keywords", []))

            for qt in query_tokens:
                if qt == name_lower:
                    score += 10
                elif name_lower.startswith(qt) or qt in name_lower:
                    score += 5
                if qt in kw_set:
                    score += 3

            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:top_k]]

    def _vector_rank(self, query: str, top_k: int) -> list[int]:
        return [i for i, _ in self._vector.rank(query, top_k)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        mode: str = "hybrid",
        per_ranker_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by the chosen retrieval mode.

        Parameters
        ----------
        query : str
            Natural-language or keyword query.
        top_k : int
            Number of chunks to return.
        mode : str
            ``"hybrid"`` (default), ``"keyword"``, or ``"vector"``.
            Hybrid falls back to keyword-only if the vector index is
            unavailable.
        per_ranker_k : int, optional
            Candidates each ranker emits before fusion. Defaults to
            ``max(top_k * 2, 10)``. Larger = more diverse fusion, minor
            compute cost.
        """
        if not self._chunks:
            return []

        mode = (mode or "hybrid").lower().strip()
        k_each = per_ranker_k if per_ranker_k is not None else max(top_k * 2, 10)

        if mode == "keyword" or not self._vector.available:
            ranked = self._keyword_rank(query, top_k)
            return [self._chunks[i] for i in ranked]

        if mode == "vector":
            ranked = self._vector_rank(query, top_k)
            return [self._chunks[i] for i in ranked]

        # hybrid
        kw = self._keyword_rank(query, k_each)
        vec = self._vector_rank(query, k_each)
        if not kw and not vec:
            return []
        if not vec:
            return [self._chunks[i] for i in kw[:top_k]]
        if not kw:
            return [self._chunks[i] for i in vec[:top_k]]
        fused = _rrf_fuse([kw, vec])
        return [self._chunks[i] for i, _ in fused[:top_k]]


class ApiQueryAgent:
    """One-shot Haiku Q&A agent for PyChrono API documentation.

    Retrieves relevant chunks via keyword search, then asks Haiku
    to answer from those chunks. Uses Anthropic prompt caching on
    the system prompt.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None,
        index_path: Path = _DEFAULT_INDEX_PATH,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model = model or getattr(
            settings, "skill_query_subagent_model", "claude-haiku-4-5-20251001"
        )
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else getattr(settings, "skill_query_subagent_max_tokens", 1024)
        )
        self.temperature = 0.0
        self.top_k = int(
            top_k if top_k is not None
            else getattr(settings, "retrieval_top_k", 5)
        )
        self.retrieval_mode = (
            retrieval_mode
            or getattr(settings, "retrieval_mode", "hybrid")
        ).lower()

        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError(
                "ApiQueryAgent: ANTHROPIC_API_KEY is not set. "
                "Set it in .env or disable API RAG."
            )
        client_kwargs: dict[str, Any] = {"api_key": key}
        base = api_base or getattr(settings, "anthropic_api_base", None)
        if base:
            client_kwargs["base_url"] = base
        self.client = anthropic.AsyncAnthropic(**client_kwargs)

        self._index = ApiChunkIndex.get(index_path)
        logger.info(
            "[ApiQueryAgent] retrieval_mode=%s vector_available=%s top_k=%d",
            self.retrieval_mode, self._index.vector_available, self.top_k,
        )

    async def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve relevant API chunks then ask Haiku to answer question.

        Returns a plain-string answer grounded in the retrieved chunks.
        """
        if not question or not question.strip():
            return "query_api requires a non-empty question."

        k = top_k if top_k is not None else self.top_k
        chunks = self._index.search(question, top_k=k, mode=self.retrieval_mode)

        if not chunks:
            return (
                "No API documentation found matching your query. "
                "The index may need rebuilding — run scripts/build_api_index.py."
            )

        # Build context block from retrieved chunks
        context_parts = [f"[Retrieved {len(chunks)} API chunk(s)]\n"]
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Chunk {i}: {chunk['id']} ---\n{chunk['text']}")
        context_text = "\n\n".join(context_parts)

        user_text = f"{context_text}\n\n---\nQuestion: {question.strip()}"

        system_blocks = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        user_blocks = [
            {
                "type": "text",
                "text": user_text,
            }
        ]

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_blocks,
                messages=[{"role": "user", "content": user_blocks}],
            )
        except Exception as exc:
            logger.warning("[ApiQueryAgent] Haiku call failed: %s", exc)
            # Fallback: return raw chunk texts so the caller still gets something
            fallback = "\n\n".join(c["text"] for c in chunks[:3])
            return f"(Haiku unavailable — raw API docs)\n\n{fallback}"

        try:
            settings = get_settings()
            if bool(getattr(settings, "log_token_usage", True)):
                usage = getattr(response, "usage", None)
                if usage is not None:
                    logger.info(
                        "[ApiQueryAgent] usage: input=%s output=%s "
                        "cache_read=%s cache_creation=%s",
                        getattr(usage, "input_tokens", 0),
                        getattr(usage, "output_tokens", 0),
                        getattr(usage, "cache_read_input_tokens", 0),
                        getattr(usage, "cache_creation_input_tokens", 0),
                    )
        except Exception:
            pass

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts).strip() or "(empty response)"
