#!/usr/bin/env python3
"""Export LFM2.5 host-side weights to a compact binary for the Swift runtime.

Reads lfm25_golden.npz (large, gitignored) and writes lfm25_host_weights.bin
(11 KB, committed to git). Swift runtime loads this at init time.

Binary layout (little-endian float32, no padding):
  Offset    Bytes   Content
  0         8192    emb_norm_weight: float32[2048]
  8192      3072    expert_bias:     float32[24][32], row-major
                    (layers 0,1 are zeros — dense, no MoE)
  Total:    11264 bytes

Run with Xcode python3 or .venv313:
  python3 scripts/export_lfm25_host_weights.py \\
    --golden models/lfm25/ane/lfm25_golden.npz \\
    --out    models/lfm25/ane/lfm25_host_weights.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

NUM_LAYERS   = 24
HIDDEN_SIZE  = 2048
NUM_EXPERTS  = 32


def export(golden_path: Path, out_path: Path) -> None:
    print(f"Loading {golden_path}  ({golden_path.stat().st_size // 1024 // 1024} MB)")
    d = np.load(str(golden_path))

    # ── emb_norm_weight ────────────────────────────────────────────────────
    if "embedding_norm_weight" not in d:
        raise SystemExit("embedding_norm_weight not found in golden npz. "
                         "Re-run validators/lfm25_golden.py --generate first.")
    emb_norm = d["embedding_norm_weight"].astype(np.float32)
    assert emb_norm.shape == (HIDDEN_SIZE,), f"unexpected shape: {emb_norm.shape}"

    # ── expert_bias (layers 2–23) ──────────────────────────────────────────
    expert_bias_all = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float32)
    loaded = 0
    for li in range(NUM_LAYERS):
        key = f"expert_bias_{li}"
        if key in d:
            bias = d[key].astype(np.float32).flatten()
            assert bias.shape == (NUM_EXPERTS,), f"layer {li}: {bias.shape}"
            expert_bias_all[li] = bias
            loaded += 1
    print(f"  emb_norm_weight:  shape={emb_norm.shape}")
    print(f"  expert_bias:      {loaded} layers loaded (layers 2–23 expected)")

    # ── Write binary ───────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(emb_norm.tobytes())                 # 8192 bytes
        f.write(expert_bias_all.tobytes())          # 3072 bytes

    total = emb_norm.nbytes + expert_bias_all.nbytes
    assert total == 11264, f"unexpected total: {total}"
    print(f"\nWrote {total} bytes → {out_path}")
    print("Verify (should print 11264):")
    print(f"  {out_path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LFM2.5 host weights")
    parser.add_argument("--golden", type=Path,
                        default=Path("models/lfm25/ane/lfm25_golden.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("models/lfm25/ane/lfm25_host_weights.bin"))
    args = parser.parse_args()

    if not args.golden.exists():
        raise SystemExit(f"Golden npz not found: {args.golden}\n"
                         "Run: python3 validators/lfm25_golden.py --generate …")
    export(args.golden, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
