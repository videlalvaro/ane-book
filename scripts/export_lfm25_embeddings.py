#!/usr/bin/env python3
"""Export LFM2.5 embed_tokens.weight to a raw float32 binary for the Swift runtime.

The embedding table (vocab_size × hidden_size = 128000 × 2048 = ~1 GB) is loaded
host-side in Swift for token embedding lookup (permitted per ANE mandate).

Output: lfm25_embeddings.bin — raw float32[128000][2048], little-endian, no header.
  Size: 128000 × 2048 × 4 = 1,048,576,000 bytes (~1 GB)

Run with any python that has safetensors:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      scripts/export_lfm25_embeddings.py \\
      --weights models/lfm25/hf/model.safetensors \\
      --out     models/lfm25/ane/lfm25_embeddings.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VOCAB_SIZE  = 128_000
HIDDEN_SIZE = 2048
EXPECTED_BYTES = VOCAB_SIZE * HIDDEN_SIZE * 4  # 1,048,576,000


def export(weights_path: Path, out_path: Path) -> None:
    import numpy as np

    # Use safetensors with PyTorch framework — handles bfloat16 transparently
    try:
        from safetensors import safe_open
        print(f"Loading from {weights_path}  ({weights_path.stat().st_size // 1024 // 1024} MB)")
        with safe_open(str(weights_path), framework="pt") as f:
            keys = list(f.keys())
            emb_key = next((k for k in keys if "embed_tokens.weight" in k), None)
            if emb_key is None:
                raise SystemExit(f"embed_tokens.weight not found. Keys: {keys[:10]}")
            print(f"Loading '{emb_key}'…")
            emb_tensor = f.get_tensor(emb_key)
            emb_f32 = emb_tensor.to(dtype=__import__("torch").float32).numpy()
    except ImportError:
        raise SystemExit("safetensors not found — run in .venv313 or install: pip install safetensors torch")

    assert emb_f32.shape == (VOCAB_SIZE, HIDDEN_SIZE), \
        f"unexpected embedding shape: {emb_f32.shape}, expected ({VOCAB_SIZE}, {HIDDEN_SIZE})"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {EXPECTED_BYTES // 1024 // 1024} MB → {out_path}…")
    emb_f32.tofile(str(out_path))

    actual = out_path.stat().st_size
    assert actual == EXPECTED_BYTES, f"size mismatch: wrote {actual}, expected {EXPECTED_BYTES}"
    print(f"Done. {actual // 1024 // 1024} MB written.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LFM2.5 embeddings to raw float32 binary")
    parser.add_argument("--weights", type=Path,
                        default=Path("models/lfm25/hf/model.safetensors"),
                        help="Path to model.safetensors")
    parser.add_argument("--out", type=Path,
                        default=Path("models/lfm25/ane/lfm25_embeddings.bin"),
                        help="Output path for raw float32 embedding binary")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    if args.out.exists():
        print(f"Already exists ({args.out.stat().st_size // 1024 // 1024} MB): {args.out}")
        print("Delete it first to re-export.")
        return 0

    export(args.weights, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
