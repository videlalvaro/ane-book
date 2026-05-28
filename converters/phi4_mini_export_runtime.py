#!/usr/bin/env python3
"""Export Phi-family runtime metadata and host embedding lookup table.

Heavy compute remains in ANE layer and LM-head shards. The exported embedding
binary is for permitted host-side token-id -> vector lookup only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONV_ANE_DIR = ROOT / "emilio" / "conv-ane"
if str(CONV_ANE_DIR) not in sys.path:
    sys.path.insert(0, str(CONV_ANE_DIR))

from gguf_to_ane import GGUFModel  # noqa: E402


DEFAULT_MODEL = ROOT / "models" / "Phi-4-mini-instruct.Q8_0.gguf"
DEFAULT_OUT_DIR = ROOT / "emilio" / "conv-ane" / "phi4_mini_ane"
DEFAULT_ARTIFACT_PREFIX = "phi4mini"
DEFAULT_LM_HEAD_PREFIX = "Phi4MiniLMHead"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Phi-family runtime manifest + embedding bin")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX,
                        help="Layer/embedding artifact prefix, e.g. phi5mini")
    parser.add_argument("--lm-head-prefix", default=DEFAULT_LM_HEAD_PREFIX,
                        help="LM-head artifact prefix, e.g. Phi5MiniLMHead")
    parser.add_argument("--layer-artifact-dir", type=Path, default=None)
    parser.add_argument("--layer-group-size", type=int, default=1)
    parser.add_argument(
        "--layer-spec",
        action="append",
        default=[],
        help="Explicit layer shard as start:end:path. May be repeated; overrides --layer-artifact-dir/--layer-group-size.",
    )
    parser.add_argument("--lm-head-dir", type=Path, default=None)
    parser.add_argument("--lm-head-num-shards", type=int, default=4)
    parser.add_argument("--include-verifier", action="store_true", help="Add T=4 speculative verifier artifacts to the manifest")
    parser.add_argument("--verifier-layer-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--verifier-layer-spec",
        action="append",
        default=[],
        help="Explicit T=4 verifier shard as start:end:path. May be repeated; used when --include-verifier is set.",
    )
    parser.add_argument("--verifier-lm-head-dir", type=Path, default=None)
    parser.add_argument("--verifier-batch-tokens", type=int, default=4)
    parser.add_argument("--manifest-name", default="phi4mini_runtime_meta.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"missing GGUF: {args.model}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gguf = GGUFModel(args.model)
    cfg = gguf.config()
    vocab = int(cfg["vocab_size"])
    d_model = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    if args.layer_group_size < 1:
        raise SystemExit("--layer-group-size must be >= 1")
    if args.lm_head_num_shards < 1:
        raise SystemExit("--lm-head-num-shards must be >= 1")
    if args.verifier_batch_tokens <= 1:
        raise SystemExit("--verifier-batch-tokens must be > 1")
    if not args.artifact_prefix or not args.lm_head_prefix:
        raise SystemExit("--artifact-prefix and --lm-head-prefix must be non-empty")

    embed_path = args.out_dir / f"{args.artifact_prefix}_token_embd_fp16.bin"
    if embed_path.exists() and not args.force:
        print(f"embedding exists; skipping export: {embed_path}")
    else:
        print(f"exporting token embedding: vocab={vocab} d={d_model}")
        embed = gguf.get_tensor("token_embd.weight", dtype=np.float16)
        if embed.shape != (vocab, d_model):
            raise SystemExit(f"embedding shape {embed.shape} != ({vocab}, {d_model})")
        embed.tofile(embed_path)
        print(f"wrote {embed_path} ({embed_path.stat().st_size / 1e9:.2f} GB)")

    layer_shards = []

    def rel_to_out_dir(path: Path) -> str:
        path = path.resolve()
        try:
            return path.relative_to(args.out_dir.resolve()).as_posix()
        except ValueError:
            return Path("..", path.relative_to(args.out_dir.resolve().parent)).as_posix()

    if args.layer_spec:
        for spec in args.layer_spec:
            parts = spec.split(":", 2)
            if len(parts) != 3:
                raise SystemExit(f"invalid --layer-spec {spec!r}; expected start:end:path")
            start = int(parts[0])
            end = int(parts[1])
            path = Path(parts[2])
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            if not path.exists():
                raise SystemExit(f"missing compiled layer shard: {path}")
            layer_shards.append({"start": start, "end": end, "path": rel_to_out_dir(path)})
    else:
        layer_dir = args.layer_artifact_dir or args.out_dir
        layer_dir = layer_dir.resolve()
        for start in range(0, n_layers, args.layer_group_size):
            end = min(start + args.layer_group_size, n_layers)
            name = f"{args.artifact_prefix}_layer{start}_{end}_q8.mlmodelc"
            path = layer_dir / name
            if not path.exists():
                raise SystemExit(f"missing compiled layer shard: {path}")
            layer_shards.append({"start": start, "end": end, "path": rel_to_out_dir(path)})

    expected_start = 0
    for shard in sorted(layer_shards, key=lambda item: item["start"]):
        if shard["start"] != expected_start or shard["end"] <= shard["start"]:
            raise SystemExit(f"invalid layer coverage at {expected_start}: got [{shard['start']},{shard['end']})")
        expected_start = shard["end"]
    if expected_start != n_layers:
        raise SystemExit(f"layer shards cover 0..{expected_start}, expected 0..{n_layers}")

    layer_shards = sorted(layer_shards, key=lambda item: item["start"])

    for shard in layer_shards:
        resolved_path = (args.out_dir / shard["path"]).resolve()
        if not resolved_path.exists():
            raise SystemExit(f"missing compiled layer shard: {resolved_path}")

    lm_head_dir = args.lm_head_dir or (args.out_dir / "lm_head_shards")
    lm_head_dir = lm_head_dir.resolve()
    lm_head_shards = []
    chunk = int(np.ceil(vocab / args.lm_head_num_shards))
    for shard in range(args.lm_head_num_shards):
        start = shard * chunk
        end = min((shard + 1) * chunk, vocab)
        path = lm_head_dir / f"{args.lm_head_prefix}_s{shard}_q8.mlmodelc"
        if not path.exists():
            raise SystemExit(f"missing LM-head shard: {path}")
        rel_name = rel_to_out_dir(path)
        lm_head_shards.append({"shard_idx": shard, "vocab_start": start, "vocab_end": end, "mlmodelc": rel_name})

    speculative_verifier = None
    if args.include_verifier:
        verifier_layers = []
        if args.verifier_layer_spec:
            for spec in args.verifier_layer_spec:
                parts = spec.split(":", 2)
                if len(parts) != 3:
                    raise SystemExit(f"invalid --verifier-layer-spec {spec!r}; expected start:end:path")
                start = int(parts[0])
                end = int(parts[1])
                path = Path(parts[2])
                if not path.is_absolute():
                    path = (ROOT / path).resolve()
                if not path.exists():
                    raise SystemExit(f"missing verifier layer shard: {path}")
                verifier_layers.append({"start": start, "end": end, "path": rel_to_out_dir(path)})
        else:
            verifier_dir = args.verifier_layer_artifact_dir or (args.out_dir / "../phi4_mini_ane_t4_verifier")
            verifier_dir = verifier_dir.resolve()
            for shard in layer_shards:
                start = int(shard["start"])
                end = int(shard["end"])
                path = verifier_dir / f"{args.artifact_prefix}_t4_layer{start}_{end}_q8.mlmodelc"
                if not path.exists():
                    raise SystemExit(f"missing verifier layer shard: {path}")
                verifier_layers.append({"start": start, "end": end, "path": rel_to_out_dir(path)})

        expected_start = 0
        for shard in sorted(verifier_layers, key=lambda item: item["start"]):
            if shard["start"] != expected_start or shard["end"] <= shard["start"]:
                raise SystemExit(f"invalid verifier layer coverage at {expected_start}: got [{shard['start']},{shard['end']})")
            expected_start = shard["end"]
        if expected_start != n_layers:
            raise SystemExit(f"verifier layer shards cover 0..{expected_start}, expected 0..{n_layers}")
        verifier_layers = sorted(verifier_layers, key=lambda item: item["start"])

        verifier_lm_head_dir = args.verifier_lm_head_dir or (args.out_dir / "lm_head_shards_bt4")
        verifier_lm_head_dir = verifier_lm_head_dir.resolve()
        verifier_lm_head_shards = []
        batch_tag = f"_bt{args.verifier_batch_tokens}"
        for shard in range(args.lm_head_num_shards):
            start = shard * chunk
            end = min((shard + 1) * chunk, vocab)
            path = verifier_lm_head_dir / f"{args.lm_head_prefix}{batch_tag}_s{shard}_q8.mlmodelc"
            if not path.exists():
                raise SystemExit(f"missing verifier LM-head shard: {path}")
            verifier_lm_head_shards.append({"shard_idx": shard, "vocab_start": start, "vocab_end": end, "mlmodelc": rel_to_out_dir(path)})

        speculative_verifier = {
            "batch_tokens": args.verifier_batch_tokens,
            "layers": verifier_layers,
            "lm_head_shards": verifier_lm_head_shards,
        }

    model_family = args.model_family or str(gguf.meta("general.name", args.model.stem))
    manifest = {
        "artifacts_version": 1,
        "model_family": model_family,
        "gguf": str(args.model),
        "d_model": d_model,
        "n_heads": int(cfg["n_heads"]),
        "n_kv_heads": int(cfg["n_kv_heads"]),
        "d_head": int(cfg["d_head"]),
        "rope_dim": int(cfg.get("rope_dim", cfg["d_head"])),
        "d_ff": int(cfg["d_ff"]),
        "vocab_size": vocab,
        "n_layers": n_layers,
        "max_seq_len": args.seq_len,
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
        "rope_freq_base": float(cfg["rope_freq_base"]),
        "eos_token_id": int(cfg["eos_token_id"]),
        "bos_token_id": int(cfg["bos_token_id"]),
        "tie_word_embeddings": "output.weight" not in gguf.tensors,
        "quant_bits": 8,
        "embed_bin": embed_path.name,
        "layers": layer_shards,
        "lm_head_shards": lm_head_shards,
    }
    if speculative_verifier is not None:
        manifest["speculative_verifier"] = speculative_verifier
    manifest_path = args.out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())