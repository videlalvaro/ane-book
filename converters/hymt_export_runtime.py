#!/usr/bin/env python3
"""Export Hunyuan runtime: embedding binary + JSON manifest.

Run with:
    /usr/bin/python3 converters/hymt_export_runtime.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "converters"))
from gguf_to_ane import GGUFModel  # noqa: E402

GGUF       = ROOT / "models" / "hymt" / "Hy-MT1.5-1.8B-2bit.gguf"
SHARD_DIR  = ROOT / "models" / "hymt" / "ane"
HEAD_DIR   = SHARD_DIR / "lm_head"
OUT_DIR    = ROOT / "models" / "hymt" / "ane"
MANIFEST   = OUT_DIR / "hymt_runtime_meta.json"
EMBED_BIN  = OUT_DIR / "hymt_token_embd_fp16.bin"

# Shard layout matching the build (5+5+5+5+5+5+2)
LAYER_SHARDS = [
    (0,  5,  SHARD_DIR / "hymt_q8_s0_5.mlmodelc"),
    (5,  10, SHARD_DIR / "hymt_q8_s5_10.mlmodelc"),
    (10, 15, SHARD_DIR / "hymt_q8_s10_15.mlmodelc"),
    (15, 20, SHARD_DIR / "hymt_q8_s15_20.mlmodelc"),
    (20, 25, SHARD_DIR / "hymt_q8_s20_25.mlmodelc"),
    (25, 30, SHARD_DIR / "hymt_q8_s25_30.mlmodelc"),
    (30, 32, SHARD_DIR / "hymt_q8_s30_32.mlmodelc"),
]
HEAD_SHARDS = [
    (0,  60409,  HEAD_DIR / "HymtLMHead_s0_q8.mlmodelc"),
    (60409, 120818, HEAD_DIR / "HymtLMHead_s1_q8.mlmodelc"),
]


def rel(path: Path) -> str:
    """Absolute path string (Swift resolvePath handles absolute paths)."""
    return str(path.resolve())


def main():
    print(f"Loading GGUF: {GGUF.name}")
    gguf = GGUFModel(str(GGUF))
    cfg  = gguf.config()
    D    = int(cfg["d_model"])
    V    = int(cfg["vocab_size"])
    arch = gguf.meta("general.architecture", "hunyuan-dense")

    # Validate shards exist
    for s, e, p in LAYER_SHARDS + HEAD_SHARDS:
        if not p.exists():
            raise SystemExit(f"Missing shard: {p}")

    # Export embedding binary (host-side lookup, policy-exempt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if EMBED_BIN.exists():
        print(f"embed binary exists: {EMBED_BIN.name}")
    else:
        print(f"Exporting embedding table: vocab={V} d={D} ...")
        emb = gguf.get_tensor("token_embd.weight", dtype=np.float16)
        assert emb.shape == (V, D), f"unexpected shape {emb.shape}"
        emb.tofile(EMBED_BIN)
        sz = EMBED_BIN.stat().st_size / 1e6
        print(f"  wrote {EMBED_BIN.name}  ({sz:.1f} MB)")

    # Build manifest
    bos = gguf.meta("tokenizer.ggml.bos_token_id", 1)
    eos = gguf.meta("tokenizer.ggml.eos_token_id", 2)
    meta = {
        "model_family": arch,
        "d_model": D,
        "n_heads": int(cfg["n_heads"]),
        "n_kv_heads": int(cfg["n_kv_heads"]),
        "d_head": int(cfg["d_head"]),
        "rope_dim": int(cfg.get("rope_dim", cfg["d_head"])),
        "vocab_size": V,
        "n_layers": int(cfg["n_layers"]),
        "max_seq_len": 1637,
        "rope_freq_base": float(cfg["rope_freq_base"]),
        "eos_token_id": int(eos),
        "bos_token_id": int(bos),
        "embed_bin": rel(EMBED_BIN),
        "layers": [
            {"start": s, "end": e, "path": rel(p)}
            for s, e, p in LAYER_SHARDS
        ],
        "lm_head_shards": [
            {"shard_idx": i, "vocab_start": s, "vocab_end": e, "mlmodelc": rel(p)}
            for i, (s, e, p) in enumerate(HEAD_SHARDS)
        ],
    }
    MANIFEST.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {MANIFEST}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
