#!/usr/bin/env python3
"""Check Phi-4-mini CoreML shard ANE residency with MLComputePlan.

Run this with Xcode's Python/coremltools 9 environment. It is generic enough for
any compiled `.mlmodelc`, but the failure policy is tuned for Phi-4-mini dense
transformer shards: all convolution/matmul-as-conv ops must prefer ANE.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CONV_OPS = {"ios18.conv", "conv", "ios18.convolution"}
IGNORED_OP_PREFIXES = ("const", "ios18.constexpr")


def is_compute_op(op_name: str) -> bool:
    """Return True for runtime compute ops that must not fall back to CPU/GPU."""
    return not any(op_name.startswith(prefix) for prefix in IGNORED_OP_PREFIXES)


def device_name(raw: str) -> str:
    if "Neural" in raw:
        return "ANE"
    if "GPU" in raw:
        return "GPU"
    if "CPU" in raw:
        return "CPU"
    return raw or "unknown"


def check_model(model_path: Path) -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    rows = []
    non_ane_compute_rows = []
    counts = Counter()
    conv_counts = Counter()
    compute_counts = Counter()
    program = plan.model_structure.program
    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = getattr(usage, "preferred_compute_device", None)
            raw = preferred.__class__.__name__ if preferred is not None else "unknown"
            dev = device_name(raw)
            op_name = getattr(op, "operator_name", "unknown")
            counts[(op_name, dev)] += 1
            if is_compute_op(op_name):
                compute_counts[dev] += 1
                if dev != "ANE":
                    non_ane_compute_rows.append({
                        "function": fn_name,
                        "op": op_name,
                        "device": dev,
                        "raw_device": raw,
                    })
            if op_name in CONV_OPS:
                conv_counts[dev] += 1
                rows.append({"function": fn_name, "op": op_name, "device": dev, "raw_device": raw})
    conv_total = sum(conv_counts.values())
    conv_ane = conv_counts.get("ANE", 0)
    compute_total = sum(compute_counts.values())
    compute_ane = compute_counts.get("ANE", 0)
    return {
        "model": str(model_path),
        "conv_total": conv_total,
        "conv_ane": conv_ane,
        "conv_non_ane": conv_total - conv_ane,
        "compute_total": compute_total,
        "compute_ane": compute_ane,
        "compute_non_ane": compute_total - compute_ane,
        "pass": conv_total > 0 and conv_total == conv_ane and compute_total == compute_ane,
        "conv_counts": dict(conv_counts),
        "compute_counts": dict(compute_counts),
        "op_device_counts": {f"{op}:{dev}": n for (op, dev), n in sorted(counts.items())},
        "conv_rows": rows,
        "non_ane_compute_rows": non_ane_compute_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to compiled .mlmodelc")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    report = check_model(args.model)
    print(f"model: {report['model']}")
    print(f"conv_total={report['conv_total']} conv_ane={report['conv_ane']} conv_non_ane={report['conv_non_ane']}")
    print(f"compute_total={report['compute_total']} compute_ane={report['compute_ane']} compute_non_ane={report['compute_non_ane']}")
    print(f"PASS={report['pass']}")
    if report["conv_non_ane"]:
        for row in report["conv_rows"]:
            if row["device"] != "ANE":
                print(f"NON_ANE_CONV function={row['function']} op={row['op']} device={row['device']} raw={row['raw_device']}")
    if report["compute_non_ane"]:
        for row in report["non_ane_compute_rows"]:
            print(f"NON_ANE_COMPUTE function={row['function']} op={row['op']} device={row['device']} raw={row['raw_device']}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"json_out: {args.json_out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
