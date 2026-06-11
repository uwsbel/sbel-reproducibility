#!/usr/bin/env python3
"""
Export a Habitat-style PointNav checkpoint to a plain state_dict checkpoint.

This is intended for using newer Habitat checkpoints with the DSInC non-habitat
PointNav loader.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, Tuple

import torch


def _bootstrap_sys_path() -> None:
    """
    Ensure local repo modules (notably `vlfm`) are importable for checkpoint unpickling.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]  # .../DSInC
    candidates = [
        repo_root,
        repo_root / "third_party" / "vlfm",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            entry = str(candidate)
            if entry not in sys.path:
                sys.path.insert(0, entry)


def _extract_state_dict(ckpt: object) -> OrderedDict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return OrderedDict(ckpt["state_dict"])
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return OrderedDict(ckpt)
    raise ValueError(
        "Unsupported checkpoint structure. Expected either a dict with 'state_dict' "
        "or a raw tensor-key dictionary."
    )


def _strip_prefix(prefix: str) -> Callable[[str], str]:
    return lambda k: k[len(prefix) :] if k.startswith(prefix) else k


def _compose(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    def inner(k: str) -> str:
        out = k
        for fn in funcs:
            out = fn(out)
        return out

    return inner


def _candidate_transforms() -> Iterable[Tuple[str, Callable[[str], str]]]:
    id_fn = lambda k: k
    strip_module = _strip_prefix("module.")
    strip_actor = _strip_prefix("actor_critic.")
    yield "identity", id_fn
    yield "strip_module", strip_module
    yield "strip_actor_critic", strip_actor
    yield "strip_module_then_actor_critic", _compose(strip_module, strip_actor)
    yield "strip_actor_critic_then_module", _compose(strip_actor, strip_module)


def _load_reference_keys() -> set[str]:
    try:
        from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
            PointNavResNetPolicy,
        )

        return set(PointNavResNetPolicy().state_dict().keys())
    except Exception:
        return set()


def _score_keys(keys: Iterable[str], reference: set[str]) -> int:
    if not reference:
        return 0
    return sum(1 for k in keys if k in reference)


def _apply_best_transform(
    state_dict: OrderedDict,
    reference_keys: set[str],
) -> Tuple[OrderedDict, str, int]:
    best_name = "identity"
    best_score = -1
    best_dict = state_dict

    for name, transform in _candidate_transforms():
        transformed = OrderedDict((transform(k), v) for k, v in state_dict.items())
        score = _score_keys(transformed.keys(), reference_keys)
        if score > best_score:
            best_score = score
            best_name = name
            best_dict = transformed

    return best_dict, best_name, best_score


def _patch_legacy_prev_action_keys(state_dict: OrderedDict) -> None:
    if "net.prev_action_embedding_cont.weight" not in state_dict and "net.prev_action_embedding.weight" in state_dict:
        state_dict["net.prev_action_embedding_cont.weight"] = state_dict["net.prev_action_embedding.weight"]
    if "net.prev_action_embedding_cont.bias" not in state_dict and "net.prev_action_embedding.bias" in state_dict:
        state_dict["net.prev_action_embedding_cont.bias"] = state_dict["net.prev_action_embedding.bias"]


def main() -> None:
    _bootstrap_sys_path()
    parser = argparse.ArgumentParser("Export PointNav checkpoint to plain state_dict format")
    parser.add_argument("--input", required=True, help="Path to source checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Path to output checkpoint (.pth)")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use torch.load(weights_only=True). Only set if source ckpt supports it.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, object] = {"map_location": "cpu"}
    if args.weights_only:
        load_kwargs["weights_only"] = True
    try:
        ckpt = torch.load(str(input_path), **load_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name == "vlfm":
            raise ModuleNotFoundError(
                "Could not import `vlfm` while loading checkpoint. "
                "Run from the repo with PYTHONPATH including `third_party/vlfm`, e.g.\n"
                "  PYTHONPATH=./third_party/vlfm:$PYTHONPATH "
                "python utils/export_pointnav_state_dict.py --input ... --output ..."
            ) from exc
        raise

    state_dict = _extract_state_dict(ckpt)
    reference_keys = _load_reference_keys()
    state_dict, transform_name, matched = _apply_best_transform(state_dict, reference_keys)
    _patch_legacy_prev_action_keys(state_dict)

    torch.save(state_dict, str(output_path))
    print(f"Saved converted checkpoint to: {output_path}")
    print(f"Input tensors: {len(state_dict)}")
    print(f"Selected key transform: {transform_name}")
    if reference_keys:
        print(f"Matched keys vs non-habitat PointNav: {matched}/{len(reference_keys)}")
    else:
        print("Reference model unavailable during export; skipped overlap scoring.")


if __name__ == "__main__":
    main()
