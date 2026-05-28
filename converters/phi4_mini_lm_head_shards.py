#!/usr/bin/env python3
"""Build ANE LM-head shards for Phi-family GGUF models.

Each shard contains final RMSNorm + tied embedding projection as Conv2d(1x1):
  hidden (1, 3072, 1, 1) -> logits slice (1, vocab_chunk, 1, 1)

Run with Xcode Python/CoreML tools. The GGUF output.weight tensor is used when
present; otherwise the token embedding is used as the tied LM head.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONV_ANE_DIR = ROOT / "emilio" / "conv-ane"
if str(CONV_ANE_DIR) not in sys.path:
    sys.path.insert(0, str(CONV_ANE_DIR))

from gguf_to_ane import GGUFModel  # noqa: E402


DEFAULT_MODEL = ROOT / "models" / "Phi-4-mini-instruct.Q8_0.gguf"
DEFAULT_OUT_DIR = ROOT / "emilio" / "conv-ane" / "phi4_mini_ane" / "lm_head_shards"
DEFAULT_ARTIFACT_PREFIX = "Phi4MiniLMHead"


def build_shard_model(norm_weight: np.ndarray, embed_slice: np.ndarray, d_model: int, vocab_start: int,
                      vocab_end: int, rms_eps: float, shard_idx: int, num_shards: int, batch_tokens: int):
    import torch
    import torch.nn as nn

    vocab_chunk = vocab_end - vocab_start

    class RMSNormConv(nn.Module):
        def __init__(self, weight: np.ndarray, eps: float):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1), requires_grad=False)

        def forward(self, x):
            k = x.shape[1] ** 0.5
            x_scaled = x * (1.0 / k)
            variance = x_scaled.pow(2).mean(dim=1, keepdim=True)
            x_normed = x_scaled * torch.rsqrt(variance + self.eps / (k * k))
            return (x_normed * self.weight).to(x.dtype)

    class PhiLMHeadShard(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNormConv(norm_weight, rms_eps)
            self.proj = nn.Conv2d(d_model, vocab_chunk, 1, bias=False)
            self.proj.weight = nn.Parameter(
                torch.tensor(embed_slice, dtype=torch.float16).reshape(vocab_chunk, d_model, 1, 1),
                requires_grad=False,
            )

        def forward(self, hidden):
            return self.proj(self.norm(hidden))

    model = PhiLMHeadShard().half().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  shard {shard_idx}/{num_shards}: vocab [{vocab_start},{vocab_end}) "
          f"= {vocab_chunk} tokens, {n_params:,} params")

    example_input = torch.randn(1, d_model, batch_tokens, 1, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    return traced, example_input


def convert_and_quantize(traced, d_model: int, batch_tokens: int, quant_bits: int):
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden", shape=(1, d_model, batch_tokens, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quant_bits == 8:
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        mlmodel = linear_quantize_weights(mlmodel, config=OptimizationConfig(global_config=op_config))
    elif quant_bits != 0:
        raise ValueError(f"unsupported quant_bits={quant_bits}; use 0 or 8")
    return mlmodel


def compile_mlpackage(pkg_path: Path) -> Path:
    out_dir = pkg_path.parent
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path.resolve()), str(out_dir.resolve())],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"coremlcompiler failed for {pkg_path}")
    mlmodelc = out_dir / f"{pkg_path.stem}.mlmodelc"
    if not mlmodelc.exists():
        raise FileNotFoundError(f"expected compiled model not found: {mlmodelc}")
    print(result.stdout.strip())
    return mlmodelc


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phi-family ANE LM head shards")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX,
                        help="Artifact filename prefix, e.g. Phi5MiniLMHead")
    parser.add_argument("--model-family", default=None,
                        help="Manifest model_family value; defaults to GGUF architecture")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-start", type=int, default=0, help="First shard index, inclusive")
    parser.add_argument("--shard-end", type=int, default=None, help="End shard index, exclusive")
    parser.add_argument("--batch-tokens", type=int, default=1, help="Number of hidden vectors scored per prediction call")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[0, 8])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing packages/models")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"missing GGUF: {args.model}")
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0")
    if args.shard_end is None:
        args.shard_end = args.num_shards
    if not (0 <= args.shard_start < args.shard_end <= args.num_shards):
        raise SystemExit("invalid --shard-start/--shard-end range")
    if args.batch_tokens <= 0:
        raise SystemExit("--batch-tokens must be > 0")
    if not args.artifact_prefix:
        raise SystemExit("--artifact-prefix must be non-empty")

    print(f"Loading GGUF metadata: {args.model}")
    gguf = GGUFModel(args.model)
    cfg = gguf.config()
    d_model = int(cfg["d_model"])
    vocab = int(cfg["vocab_size"])
    rms_eps = float(cfg["rms_norm_eps"])
    if "token_embd.weight" not in gguf.tensors or "output_norm.weight" not in gguf.tensors:
        raise SystemExit("missing token_embd.weight or output_norm.weight")
    lm_head_tensor = "output.weight" if "output.weight" in gguf.tensors else "token_embd.weight"
    tied_embedding = lm_head_tensor == "token_embd.weight"

    print(f"  vocab={vocab} d_model={d_model} eps={rms_eps} shards={args.num_shards} quant={args.quant_bits}")
    print(f"  lm_head_tensor={lm_head_tensor} tied_embedding={tied_embedding}")
    chunk_size = math.ceil(vocab / args.num_shards)
    shard_ranges = [(i * chunk_size, min((i + 1) * chunk_size, vocab)) for i in range(args.num_shards)]
    for i, (start, end) in enumerate(shard_ranges):
        print(f"  shard {i}: vocab [{start},{end}) tokens={end-start} int8≈{(end-start)*d_model/1e6:.1f} MB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_tag = "q8" if args.quant_bits == 8 else "fp16"
    batch_tag = "" if args.batch_tokens == 1 else f"_bt{args.batch_tokens}"

    print("Loading LM-head weight + final norm from GGUF...")
    lm_head_weight = gguf.get_tensor(lm_head_tensor, dtype=np.float16)
    norm = gguf.get_tensor("output_norm.weight", dtype=np.float16)
    if lm_head_weight.shape != (vocab, d_model):
        raise SystemExit(f"LM-head shape {lm_head_weight.shape} != ({vocab}, {d_model})")
    if norm.shape != (d_model,):
        raise SystemExit(f"output_norm shape {norm.shape} != ({d_model},)")

    built = []
    t0 = time.time()
    for shard_idx in range(args.shard_start, args.shard_end):
        start, end = shard_ranges[shard_idx]
        name = f"{args.artifact_prefix}{batch_tag}_s{shard_idx}_{q_tag}"
        pkg_path = args.output_dir / f"{name}.mlpackage"
        mlmodelc_path = args.output_dir / f"{name}.mlmodelc"
        print(f"\n{'=' * 60}\nBuilding {name} vocab [{start},{end})")

        if mlmodelc_path.exists() and not args.force:
            print(f"  compiled model exists; skipping: {mlmodelc_path}")
            built.append({"shard_idx": shard_idx, "vocab_start": start, "vocab_end": end,
                          "mlpackage": pkg_path.name if pkg_path.exists() else None,
                          "mlmodelc": mlmodelc_path.name,
                          "skipped": True})
            continue
        if pkg_path.exists() and not args.force and not args.no_compile:
            print(f"  package exists; compiling only: {pkg_path}")
            mlmodelc = compile_mlpackage(pkg_path)
            built.append({"shard_idx": shard_idx, "vocab_start": start, "vocab_end": end,
                          "mlpackage": pkg_path.name, "mlmodelc": mlmodelc.name,
                          "pkg_size_mb": dir_size_mb(pkg_path), "skipped": False})
            continue

        traced, _ = build_shard_model(norm, lm_head_weight[start:end], d_model, start, end, rms_eps, shard_idx, args.num_shards, args.batch_tokens)
        mlmodel = convert_and_quantize(traced, d_model, args.batch_tokens, args.quant_bits)
        mlmodel.save(str(pkg_path))
        pkg_mb = dir_size_mb(pkg_path)
        print(f"  saved {pkg_path} ({pkg_mb:.1f} MB)")
        mlmodelc_name = None
        if not args.no_compile:
            mlmodelc = compile_mlpackage(pkg_path)
            mlmodelc_name = mlmodelc.name
            print(f"  compiled {mlmodelc} ({dir_size_mb(mlmodelc):.1f} MB)")
        built.append({"shard_idx": shard_idx, "vocab_start": start, "vocab_end": end,
                      "mlpackage": pkg_path.name, "mlmodelc": mlmodelc_name,
                      "pkg_size_mb": pkg_mb, "skipped": False})

    model_family = args.model_family or str(gguf.meta("general.name", args.model.stem))
    manifest = {
        "model_family": model_family,
        "num_shards": args.num_shards,
        "built_shard_start": args.shard_start,
        "built_shard_end": args.shard_end,
        "vocab_size": vocab,
        "d_model": d_model,
        "batch_tokens": args.batch_tokens,
        "rms_norm_eps": rms_eps,
        "softcap": 0.0,
        "quant": "int8" if args.quant_bits == 8 else "fp16",
        "lm_head_tensor": lm_head_tensor,
        "tied_embedding": tied_embedding,
        "shards": [{
            "shard_idx": i,
            "vocab_start": s,
            "vocab_end": e,
            "mlpackage": f"{args.artifact_prefix}{batch_tag}_s{i}_{q_tag}.mlpackage",
            "mlmodelc": f"{args.artifact_prefix}{batch_tag}_s{i}_{q_tag}.mlmodelc" if (args.output_dir / f"{args.artifact_prefix}{batch_tag}_s{i}_{q_tag}.mlmodelc").exists() else None,
        } for i, (s, e) in enumerate(shard_ranges)],
    }
    manifest_path = args.output_dir / ("lm_head_manifest.json" if args.batch_tokens == 1 else f"lm_head_bt{args.batch_tokens}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSUMMARY elapsed={time.time() - t0:.1f}s built={len(built)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())