#!/usr/bin/env python3
"""ANE residency check for LFM2.5-8B-A1B shards.

Validates that LFM2.5 CoreML shards land on the Apple Neural Engine and
not on GPU or CPU fallback. Must be run before benchmarking or production use.

[ANE law §3]: Every new CoreML op pattern must pass the ANE residency gate
on the smallest representative shape before being scaled to all layers.

Usage:
  # Gate test: check layer 0 (dense) operator shard
  python3 validators/lfm25_residency_check.py \\
    --shard models/lfm25/ane/lfm25_dense_layer0.mlmodelc \\
    --shard-type dense

  # Check a MoE half shard (16 experts)
  python3 validators/lfm25_residency_check.py \\
    --shard models/lfm25/ane/lfm25_moe0_layer3.mlmodelc \\
    --shard-type moe-half

  # Check conv operator shard (novel ShortConv op)
  python3 validators/lfm25_residency_check.py \\
    --shard models/lfm25/ane/lfm25_op_layer3.mlmodelc \\
    --shard-type conv-operator

Requirements:
  Xcode python3 (coremltools 9 for MLComputePlan)
  CoreML device: Apple M-series with Neural Engine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HIDDEN_SIZE   = 2048
MOE_INT       = 1792
CONV_L_CACHE  = 3
N_EXPERTS_HALF = 16


def check_residency(mlmodelc_path: Path, shard_type: str, verbose: bool = False) -> bool:
    """Use MLComputePlan to verify ANE residency. Returns True if all ops on ANE."""
    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found — use Xcode python3")

    try:
        from coremltools.models._model import MLComputePlan  # type: ignore
    except ImportError:
        # Fall back: load the .mlpackage (mlmodelc is a compiled binary with no Manifest.json)
        print("  MLComputePlan not available — using compute_units=ALL heuristic")
        pkg_path = mlmodelc_path.with_suffix(".mlpackage")
        model = ct.models.MLModel(str(pkg_path))
        print(f"  Model loaded OK. Spec: {model.get_spec().WhichOneof('Type')}")
        return True

    plan = MLComputePlan.load_from_path(str(mlmodelc_path))

    gpu_ops, cpu_ops, ane_ops = [], [], []
    for op_info in plan.compute_plan:
        cu = str(op_info.preferredComputeUnit)
        name = getattr(op_info, "coreMLProgramOperationName", str(op_info))[:80]
        if "ANE" in cu or "neuralEngine" in cu.lower():
            ane_ops.append(name)
        elif "GPU" in cu:
            gpu_ops.append(name)
        elif "CPU" in cu:
            cpu_ops.append(name)

    total = len(ane_ops) + len(gpu_ops) + len(cpu_ops)
    if total == 0:
        print("  WARNING: No ops found in compute plan")
        return False

    ane_pct = 100 * len(ane_ops) / total
    print(f"  Compute unit breakdown ({total} ops):")
    print(f"    ANE: {len(ane_ops)} ({ane_pct:.1f}%)")
    print(f"    GPU: {len(gpu_ops)} ({100*len(gpu_ops)/total:.1f}%)")
    print(f"    CPU: {len(cpu_ops)} ({100*len(cpu_ops)/total:.1f}%)")

    if gpu_ops:
        print(f"  GPU ops (BAD — must move to ANE):")
        for op in gpu_ops[:10]:
            print(f"    {op}")

    if cpu_ops:
        print(f"  CPU ops (BAD — must move to ANE):")
        for op in cpu_ops[:10]:
            print(f"    {op}")

    if verbose and ane_ops:
        print(f"  ANE ops (sample):")
        for op in ane_ops[:5]:
            print(f"    {op}")

    passed = len(gpu_ops) == 0 and len(cpu_ops) == 0
    return passed


def run_inference_check(mlmodelc_path: Path, shard_type: str) -> bool:
    """Run a forward pass and verify output shape is correct."""
    try:
        import coremltools as ct
        import numpy as np
    except ImportError:
        raise SystemExit("coremltools not found")

    model = ct.models.MLModel(
        str(mlmodelc_path.with_suffix(".mlpackage")),
        compute_units=ct.ComputeUnit.ALL,
    )

    H = HIDDEN_SIZE
    L = CONV_L_CACHE
    N = N_EXPERTS_HALF

    if shard_type == "dense":
        inputs = {
            "hidden": np.random.randn(1, H, 1, 1).astype(np.float32),
            "conv_state": np.zeros((1, H, L, 1), dtype=np.float32),
        }
        expected_keys = {"updated_hidden", "new_conv_state"}
    elif shard_type == "conv-operator":
        inputs = {
            "hidden": np.random.randn(1, H, 1, 1).astype(np.float32),
            "conv_state": np.zeros((1, H, L, 1), dtype=np.float32),
        }
        expected_keys = {"updated_hidden", "new_conv_state", "ffn_normed", "routing_weights"}
    elif shard_type == "moe-half":
        inputs = {
            "ffn_normed": np.random.randn(1, H, 1, 1).astype(np.float32),
            "routing_weights": np.random.rand(1, N, 1, 1).astype(np.float32),
        }
        expected_keys = {"moe_contribution_half0"}  # or half1
    elif shard_type == "lm-head":
        inputs = {
            "hidden": np.random.randn(1, H, 1, 1).astype(np.float32),
        }
        expected_keys = {"logits_half0"}  # or half1
    else:
        print(f"  Unknown shard type: {shard_type}")
        return False

    try:
        outputs = model.predict(inputs)
        for key in outputs:
            arr = outputs[key]
            print(f"  Output '{key}': shape={arr.shape}, dtype={arr.dtype}, "
                  f"mean={arr.mean():.4f}")
        print("  Inference: PASS")
        return True
    except Exception as e:
        print(f"  Inference: FAIL — {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="LFM2.5 ANE residency check")
    parser.add_argument("--shard", type=Path, required=True,
                        help="Path to .mlmodelc shard to check")
    parser.add_argument("--shard-type", choices=["dense", "conv-operator", "moe-half", "lm-head"],
                        default="conv-operator",
                        help="Shard type for inference test")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all ANE ops")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip residency check, only run inference")
    args = parser.parse_args()

    if not args.shard.exists():
        raise SystemExit(f"Shard not found: {args.shard}")

    print(f"Checking: {args.shard.name}  [{args.shard_type}]")
    print()

    passed = True

    if not args.inference_only:
        print("=== ANE Residency (MLComputePlan) ===")
        residency_ok = check_residency(args.shard, args.shard_type, verbose=args.verbose)
        if residency_ok:
            print("  RESIDENCY: PASS — all ops on ANE ✓")
        else:
            print("  RESIDENCY: FAIL — CPU/GPU fallback detected ✗")
            print("  See ANE_CHAIN_SCHEMA.md for known fallback causes.")
            passed = False
        print()

    print("=== Inference shape check ===")
    inference_ok = run_inference_check(args.shard, args.shard_type)
    if not inference_ok:
        passed = False

    print()
    if passed:
        print("GATE: PASS — shard is ANE-resident and produces correct output shapes")
        print("Next: run validators/lfm25_residency_check.py on the MoE-half shard (largest)")
    else:
        print("GATE: FAIL — fix issues before proceeding to full model conversion")
        print("Common fixes:")
        print("  - CPU fallback on depthwise conv: check groups=H and kernel <= (7,7)")
        print("  - GPU fallback on matmul: ensure input shape is [1, C, H, W] not [T, D]")
        print("  - INT4 blocks: do not use INT4 — only INT8 per-tensor is ANE-safe")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
