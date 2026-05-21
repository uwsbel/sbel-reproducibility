"""Build a flat, searchable PyChrono API chunk index.

Reads data/pychrono_docs/pychrono_api.json and writes:
  - data/pychrono_docs/api_chunks.json    — per-class/function chunks
  - data/pychrono_docs/api_embeddings.npy — L2-normalized float16 matrix
                                             (when --embed, default on)
  - data/pychrono_docs/api_embeddings.meta.json — model name / dim / count

The JSON feeds keyword retrieval; the .npy feeds semantic retrieval.
api_rag.ApiChunkIndex reads both and does hybrid ranking.

Usage:
    python scripts/build_api_index.py                  # build JSON + embeddings
    python scripts/build_api_index.py --no-embed       # skip embeddings (fast)
    python scripts/build_api_index.py --embed-model MODEL
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Module → import alias used in PyChrono Python bindings
_MODULE_ALIAS = {
    "core": "pychrono",
    "vehicle": "pychrono.vehicle",
    "sensor": "pychrono.sensor",
    "fea": "pychrono.fea",
    "robot": "pychrono.robot",
    "irrlicht": "pychrono.irrlicht",
    "cascade": "pychrono.cascade",
    "fsi": "pychrono.fsi",
    "postprocess": "pychrono.postprocess",
    "parsers": "pychrono.parsers",
    "pardisomkl": "pychrono.pardisomkl",
}

# Skip these top-level keys — they're meta or duplicates
_SKIP_MODULES = {"_metadata", "classes", "decorators", "functions"}


def _clean(text: str | None, max_chars: int = 300) -> str:
    if not text:
        return ""
    text = text.strip()
    # Collapse internal newlines in docstrings to single space
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text[:max_chars]


def _keywords_for(module: str, name: str, entry: dict[str, Any]) -> list[str]:
    """Return lowercase keyword tokens for fast retrieval scoring."""
    tokens: set[str] = set()
    tokens.add(name.lower())
    tokens.add(module.lower())
    # Split CamelCase → individual words
    parts = re.sub(r"([A-Z][a-z]+)", r" \1", name).split()
    tokens.update(p.lower() for p in parts)
    # Add method names
    for mname in entry.get("methods", {}).keys():
        tokens.add(mname.lower())
        mparts = re.sub(r"([A-Z][a-z]+)", r" \1", mname).split()
        tokens.update(p.lower() for p in mparts)
    # Add words from description
    desc = entry.get("description", "")
    desc_words = re.findall(r"[a-zA-Z]{3,}", desc)
    tokens.update(w.lower() for w in desc_words[:30])
    return sorted(tokens)


def _format_class_chunk(module: str, name: str, entry: dict[str, Any]) -> str:
    """Render a class entry as a compact text block for Haiku consumption."""
    alias = _MODULE_ALIAS.get(module, f"pychrono.{module}")
    lines: list[str] = [f"## {alias}.{name}"]

    sig = entry.get("signature", "")
    if sig:
        lines.append(f"Constructor: {sig}")

    desc = _clean(entry.get("description", ""), max_chars=400)
    if desc:
        lines.append(f"Description: {desc}")

    methods: dict[str, Any] = entry.get("methods", {})
    if methods:
        lines.append("Methods:")
        for mname, minfo in methods.items():
            if not isinstance(minfo, dict):
                continue
            msig = minfo.get("signature", mname)
            mdesc = _clean(minfo.get("description", ""), max_chars=120)
            line = f"  {msig}"
            if mdesc:
                line += f"  — {mdesc}"
            lines.append(line)

    return "\n".join(lines)


def _format_function_chunk(module: str, name: str, entry: dict[str, Any]) -> str:
    alias = _MODULE_ALIAS.get(module, f"pychrono.{module}")
    lines = [f"## {alias}.{name} (function)"]
    sig = entry.get("signature", "")
    if sig:
        lines.append(f"Signature: {sig}")
    desc = _clean(entry.get("description", ""), max_chars=300)
    if desc:
        lines.append(f"Description: {desc}")
    return "\n".join(lines)


def build_chunks(api_data: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for module, entries in api_data.items():
        if module in _SKIP_MODULES:
            continue
        if not isinstance(entries, dict):
            continue

        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue

            etype = entry.get("type", "class")
            chunk_id = f"{module}.{name}"

            if etype == "class":
                text = _format_class_chunk(module, name, entry)
            elif etype == "function":
                text = _format_function_chunk(module, name, entry)
            else:
                # constant / enum value — compact one-liner
                value = entry.get("value", "")
                desc = _clean(entry.get("docstring", entry.get("description", "")), 80)
                text = (
                    f"## {module}.{name} = {value}"
                    + (f"  ({desc})" if desc else "")
                )

            chunks.append({
                "id": chunk_id,
                "module": module,
                "name": name,
                "type": etype,
                "keywords": _keywords_for(module, name, entry),
                "text": text,
            })

    return chunks


def _embed_text_for(chunk: dict[str, Any]) -> str:
    """Return the string to feed the embedding model for a given chunk.

    Prepends the qualified name so the vector space separates e.g.
    ``core.ChBody`` from ``vehicle.ChBody`` even when their descriptions
    are similar, and truncates the method block so a single monster class
    (like `core.ChBody` with 200+ methods) doesn't dominate the context
    window of the encoder.
    """
    header = f"{chunk['module']}.{chunk['name']}"
    body = chunk.get("text", "")
    # MiniLM-L6 has a 256 wordpiece context; beyond ~1200 chars the
    # encoder truncates anyway, so cap here to keep batching fast.
    if len(body) > 1200:
        body = body[:1200]
    return f"{header}\n{body}"


def build_embeddings(
    chunks: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode every chunk with the given sentence-transformers model.

    Returns ``(matrix_float16, metadata_dict)``. The matrix is L2-normalized
    in float32 before casting down to float16 for on-disk storage.
    """
    # Lazy import so --no-embed path doesn't need the dep.
    from chrono_agent.tools.embeddings import encode, get_embedding_dim

    dim = get_embedding_dim(model_name)
    print(f"Embedding {len(chunks)} chunks with {model_name} (dim={dim})...")
    texts = [_embed_text_for(c) for c in chunks]
    t0 = time.time()
    matrix = encode(texts, model_name=model_name, batch_size=batch_size, show_progress=True)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(chunks) / max(elapsed, 1e-9):.1f} chunks/s)")

    meta = {
        "model": model_name,
        "dim": dim,
        "count": len(chunks),
        "dtype": "float16",
        "normalized": True,
        "built_at": int(time.time()),
    }
    return matrix.astype(np.float16, copy=False), meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PyChrono API chunk index.")
    parser.add_argument(
        "--src",
        default="data/pychrono_docs/pychrono_api.json",
        help="Source API JSON (default: data/pychrono_docs/pychrono_api.json)",
    )
    parser.add_argument(
        "--out",
        default="data/pychrono_docs/api_chunks.json",
        help="Output chunk file (default: data/pychrono_docs/api_chunks.json)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation (JSON only). Faster when iterating on chunk format.",
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {src} ({src.stat().st_size // 1024} KB)...")
    with src.open(encoding="utf-8") as f:
        api_data = json.load(f)

    chunks = build_chunks(api_data)

    # Stats
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    print(f"Built {len(chunks)} chunks: {dict(sorted(by_type.items()))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, separators=(",", ":"))

    out_kb = out.stat().st_size // 1024
    print(f"Written to {out} ({out_kb} KB)")

    if args.no_embed:
        print("Skipping embeddings (--no-embed). Vector retrieval will be disabled.")
        return

    matrix, meta = build_embeddings(
        chunks, model_name=args.embed_model, batch_size=args.embed_batch_size,
    )
    npy_path = out.with_name("api_embeddings.npy")
    meta_path = out.with_name("api_embeddings.meta.json")
    np.save(npy_path, matrix)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Written {npy_path.name} ({npy_path.stat().st_size // 1024} KB, "
        f"shape={matrix.shape}, dtype={matrix.dtype}) and {meta_path.name}"
    )


if __name__ == "__main__":
    main()
