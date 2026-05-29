#!/usr/bin/env python3
"""Build RangeDim (T=1..4) stateful layer shards for HyMT 1.8B ANE.

A single compiled .mlmodelc handles both:
  - T=1 greedy decode (one real token)
  - T=4 chunked prefill / block verification

Key differences from phi4_mini_rangedim_export_shard.py:
  - HyMT architecture: d_model=2048, n_heads=16, n_kv_heads=4, d_head=128,
    rope_dim=128, d_ff=6144, max_seq_len=512
  - HyMT has per-head Q/K RMSNorm (has_qk_norm=True).  Applied after QKV
    projection, before RoPE.  Implemented T-agnostically by chunking the
    channel dim into n_heads groups (each [1, dh, T, 1]) and normalising
    independently — no T-dependent reshapes.
  - Same KV-scatter matmul trick as Phi4: replaces per-T for-loop so the
    traced graph contains no T-dependent control flow.

Source: Iverson APL inner-product principle (BOOK_ANALYSIS.md Exp 26)
Run with Xcode python3 (coremltools 9):
  /usr/bin/python3 python/hymt_rangedim_export_shard.py

Build a single pilot shard first:
  /usr/bin/python3 python/hymt_rangedim_export_shard.py --layer-start 30 --layer-end 32
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "converters"))
from gguf_to_ane import GGUFModel  # noqa: E402

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
)

GGUF_PATH   = ROOT / "models" / "hymt" / "Hy-MT1.5-1.8B-2bit.gguf"
OUTPUT_DIR  = ROOT / "models" / "hymt" / "ane" / "rangedim"
BASE_META   = ROOT / "models" / "hymt" / "hymt_runtime_meta.json"
OUT_META    = ROOT / "models" / "hymt" / \
              "hymt_runtime_meta_rangedim.json"

TRACE_T     = 1       # trace at T=1 (greedy default)
T_MAX       = 4       # RangeDim upper bound; enables T=1..4 at runtime
MAX_SEQ     = 512     # HyMT max sequence length
QUANT_BITS  = 8

# Shard topology: 6×5 + 1×2 = 32 layers
SHARD_RANGES = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 32)]


# ─── Model definition ────────────────────────────────────────────────────────

def build_rangedim_shard(gguf, cfg, layer_start: int, layer_end: int) -> nn.Module:
    """Return a PyTorch T=TRACE_T stateful shard covering layers [layer_start, layer_end).

    QK norm (HyMT-specific) is applied T-agnostically by chunking the channel
    dimension into n_heads/n_kv_heads groups (each [1, dh, T, 1]) and computing
    per-group RMSNorm.  No T-dependent reshapes are introduced.
    """
    d         = cfg["d_model"]      # 2048
    nh        = cfg["n_heads"]      # 16
    nkv       = cfg["n_kv_heads"]   # 4
    dh        = cfg["d_head"]       # 128
    dff       = cfg["d_ff"]         # 6144
    eps       = cfg["rms_norm_eps"]
    rope_dim  = cfg.get("rope_dim", dh)   # 128
    rope_half = rope_dim // 2             # 64
    hpk       = nh // nkv                 # 4
    kv_dim    = nkv * dh                  # 512
    qkv_dim   = d + 2 * kv_dim           # 3072

    class RMSNorm(nn.Module):
        def __init__(self, weight, eps_val):
            super().__init__()
            self.eps = eps_val
            self.w = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)
        def forward(self, x):   # x: [1, d, T, 1]
            K = x.shape[1] ** 0.5
            xs = x * (1.0 / K)
            v = xs.pow(2).mean(dim=1, keepdim=True)
            return (xs * torch.rsqrt(v + self.eps / (K * K)) * self.w).half()

    class HeadRMSNorm(nn.Module):
        """T-agnostic per-head RMSNorm used for HyMT Q/K normalization.

        Input:  [1, n_heads*d_head, T, 1]
        Output: [1, n_heads*d_head, T, 1]

        Implementation: chunk channel dim into n_heads groups of d_head
        channels, apply RMSNorm over the d_head axis for each group.
        The loop is over n_heads (static at trace time) — no T-dependent
        control flow.
        """
        def __init__(self, weight, n_heads, d_head, eps_val):
            super().__init__()
            self.eps = eps_val
            self.n_heads = n_heads
            self.d_head = d_head
            w = np.asarray(weight, dtype=np.float16).flatten()
            # Tile weights to (n_heads * d_head, 1, 1) — same weight per head
            w_tiled = np.tile(w, n_heads)
            self.w = nn.Parameter(
                torch.tensor(w_tiled, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # x: [1, n_heads*dh, T, 1]
            # Split channel into n_heads chunks of [1, dh, T, 1]
            # mean(dim=1) over each chunk → [1, 1, T, 1] — T-agnostic
            chunks = x.chunk(self.n_heads, dim=1)
            normed = []
            K = self.d_head ** 0.5
            for chunk in chunks:  # each: [1, dh, T, 1]
                cs = chunk * (1.0 / K)
                v = cs.pow(2).mean(dim=1, keepdim=True)  # [1, 1, T, 1]
                normed.append(cs * torch.rsqrt(v + self.eps / (K * K)))
            x_norm = torch.cat(normed, dim=1)  # [1, n_heads*dh, T, 1]
            return (x_norm * self.w).half()

    class RangeDimLayer(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            p = f"blk.{layer_idx}"

            self.attn_norm = RMSNorm(gguf.get_tensor(f"{p}.attn_norm.weight"), eps)

            q_w, k_w, v_w = gguf.get_qkv_weights(p, cfg)
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            has_b = gguf.has_biases(p)
            self.qkv = nn.Conv2d(d, qkv_dim, 1, bias=has_b)
            self.qkv.weight = nn.Parameter(
                torch.tensor(qkv_w, dtype=torch.float16).reshape(qkv_dim, d, 1, 1),
                requires_grad=False)
            if has_b:
                b0, b1, b2 = gguf.get_qkv_biases(p, cfg)
                self.qkv.bias = nn.Parameter(
                    torch.tensor(np.concatenate([b0, b1, b2]), dtype=torch.float16),
                    requires_grad=False)

            o_w = gguf.get_tensor(f"{p}.attn_output.weight")
            self.out = nn.Conv2d(d, d, 1, bias=False)
            self.out.weight = nn.Parameter(
                torch.tensor(o_w, dtype=torch.float16).reshape(d, d, 1, 1),
                requires_grad=False)

            # Per-head Q/K RMSNorm (HyMT-specific)
            q_norm_w = gguf.get_tensor(f"{p}.attn_q_norm.weight")
            k_norm_w = gguf.get_tensor(f"{p}.attn_k_norm.weight")
            self.q_norm = HeadRMSNorm(q_norm_w, nh, dh, eps)
            self.k_norm = HeadRMSNorm(k_norm_w, nkv, dh, eps)

            self.ffn_norm = RMSNorm(gguf.get_tensor(f"{p}.ffn_norm.weight"), eps)
            gate_w, up_w = gguf.get_gate_up_weights(p, cfg)
            gu_w = np.concatenate([gate_w, up_w], axis=0)
            self.gate_up = nn.Conv2d(d, 2 * dff, 1, bias=False)
            self.gate_up.weight = nn.Parameter(
                torch.tensor(gu_w, dtype=torch.float16).reshape(2 * dff, d, 1, 1),
                requires_grad=False)
            dn_w = gguf.get_tensor(f"{p}.ffn_down.weight")
            self.down = nn.Conv2d(dff, d, 1, bias=False)
            self.down.weight = nn.Parameter(
                torch.tensor(dn_w, dtype=torch.float16).reshape(d, dff, 1, 1),
                requires_grad=False)

            self.register_buffer("k_state",
                torch.zeros(1, nkv, MAX_SEQ, dh, dtype=torch.float16))
            self.register_buffer("v_state",
                torch.zeros(1, nkv, MAX_SEQ, dh, dtype=torch.float16))

        def forward(self, x, rope_cos, rope_sin, attn_mask, kv_write_mask):
            """
            x:              [1, d, T, 1]
            rope_cos:       [T, rope_half]
            rope_sin:       [T, rope_half]
            attn_mask:      [1, 1, T, S]
            kv_write_mask:  [1, 1, S, T]
            """
            residual = x
            nx = self.attn_norm(x)      # [1, d, T, 1]

            # ── QKV projection ───────────────────────────────────────────
            qkv = self.qkv(nx)          # [1, qkv_dim, T, 1]
            q_raw = qkv[:, :d, :, :]                    # [1, d, T, 1]
            k_raw = qkv[:, d:d + kv_dim, :, :]          # [1, kv_dim, T, 1]
            v_raw = qkv[:, d + kv_dim:, :, :]           # [1, kv_dim, T, 1]

            # ── Per-head Q/K RMSNorm (before RoPE) ──────────────────────
            q_normed = self.q_norm(q_raw)    # [1, d, T, 1]
            k_normed = self.k_norm(k_raw)    # [1, kv_dim, T, 1]

            # ── RoPE ─────────────────────────────────────────────────────
            def apply_rope(xc, n_heads):
                # xc: [1, n_heads*dh, T, 1]
                xc_sq = xc.squeeze(-1).permute(2, 0, 1)    # [T, 1, n_heads*dh]
                T_val = xc_sq.shape[0]
                xc_sq = xc_sq.reshape(T_val, n_heads, dh)  # [T, n_heads, dh]
                rot   = xc_sq[:, :, :rope_dim]
                pass_ = xc_sq[:, :, rope_dim:]
                lo    = rot[:, :, :rope_half]
                hi    = rot[:, :, rope_half:]
                c = rope_cos.unsqueeze(1)                   # [T, 1, rope_half]
                s = rope_sin.unsqueeze(1)
                rot_out = torch.cat([lo * c - hi * s, lo * s + hi * c], dim=-1)
                out = torch.cat([rot_out, pass_], dim=-1)   # [T, n_heads, dh]
                # Permute to [n_heads, dh, T] then reshape → [1, n_heads*dh, T, 1]
                T_out = out.shape[0]
                out = out.permute(1, 2, 0).reshape(1, n_heads * dh, T_out, 1)
                return out

            q = apply_rope(q_normed, nh)    # [1, d, T, 1]
            k = apply_rope(k_normed, nkv)   # [1, kv_dim, T, 1]
            v = v_raw                        # [1, kv_dim, T, 1]

            # ── KV scatter (matmul replaces for-loop) ─────────────────────
            T_cur = v.shape[2]
            new_k = k.squeeze(-1).permute(0, 2, 1).reshape(1, T_cur, nkv, dh).permute(0, 2, 1, 3)
            new_v = v.squeeze(-1).permute(0, 2, 1).reshape(1, T_cur, nkv, dh).permute(0, 2, 1, 3)
            k_written = torch.matmul(kv_write_mask, new_k)   # [1, nkv, S, dh]
            v_written = torch.matmul(kv_write_mask, new_v)
            write_any = kv_write_mask.sum(dim=-1, keepdim=True)
            k_updated = self.k_state * (1.0 - write_any) + k_written
            v_updated = self.v_state * (1.0 - write_any) + v_written
            self.k_state[:] = k_updated
            self.v_state[:] = v_updated

            # ── GQA attention ─────────────────────────────────────────────
            T_q = q.shape[2]
            q_h = q.squeeze(-1).permute(0, 2, 1).reshape(1, T_q, nh, dh).permute(0, 2, 1, 3)
            attn_parts = []
            scale = dh ** -0.5
            for kvi in range(nkv):
                q_g = q_h[:, kvi * hpk:(kvi + 1) * hpk, :, :]
                k_h = k_updated[:, kvi:kvi + 1, :, :]
                v_h = v_updated[:, kvi:kvi + 1, :, :]
                k_t = k_h.transpose(-2, -1)
                sc  = torch.matmul(q_g, k_t) * scale + attn_mask
                aw  = torch.softmax(sc.float(), dim=-1).half()
                attn_parts.append(torch.matmul(aw, v_h))
            ao = torch.cat(attn_parts, dim=1)              # [1, nh, T, dh]
            T_ao = ao.shape[2]
            ao = ao.permute(0, 1, 3, 2).reshape(1, d, T_ao, 1)
            ao = self.out(ao)
            x = residual + ao

            # ── FFN ───────────────────────────────────────────────────────
            r2 = x
            nx2 = self.ffn_norm(x)
            gu  = self.gate_up(nx2)
            gate = gu[:, :dff, :, :]
            up   = gu[:, dff:, :, :]
            h = F.silu(gate.float()).half() * up
            return r2 + self.down(h)

    class RangeDimShard(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                RangeDimLayer(i) for i in range(layer_start, layer_end)
            ])
        def forward(self, x, rope_cos, rope_sin, attn_mask, kv_write_mask):
            for layer in self.layers:
                x = layer(x, rope_cos, rope_sin, attn_mask, kv_write_mask)
            return x

    return RangeDimShard()


# ─── Build + compile one shard ───────────────────────────────────────────────

def export_shard(gguf_path: Path, cfg: dict, layer_start: int, layer_end: int,
                 output_dir: Path, name: str) -> Path:
    print(f"\n{'='*60}")
    print(f"Building RangeDim shard [{layer_start},{layer_end})  →  {name}")
    print(f"  Trace T={TRACE_T}, RangeDim T∈[1..{T_MAX}]")
    print(f"{'='*60}")

    gguf = GGUFModel(str(gguf_path))

    d         = cfg["d_model"]
    nkv       = cfg["n_kv_heads"]
    dh        = cfg["d_head"]
    rope_dim  = cfg.get("rope_dim", dh)
    rope_half = rope_dim // 2

    model = build_rangedim_shard(gguf, cfg, layer_start, layer_end)
    model.half().eval()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    T = TRACE_T
    x_ex   = torch.randn(1, d, T, 1, dtype=torch.float16)
    cos_ex = torch.randn(T, rope_half, dtype=torch.float16)
    sin_ex = torch.randn(T, rope_half, dtype=torch.float16)
    mask_ex = torch.full((1, 1, T, MAX_SEQ), -1e4, dtype=torch.float16)
    mask_ex[:, :, :T, :T] = 0
    wm_ex  = torch.zeros(1, 1, MAX_SEQ, T, dtype=torch.float16)
    for t in range(T):
        wm_ex[0, 0, t, t] = 1.0

    with torch.no_grad():
        out = model(x_ex, cos_ex, sin_ex, mask_ex, wm_ex)
        print(f"  trace output shape: {out.shape}")
    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wm_ex))

    T_dim      = ct.RangeDim(lower_bound=1, upper_bound=T_MAX, default=TRACE_T)
    rope_T_dim = ct.RangeDim(lower_bound=1, upper_bound=T_MAX, default=TRACE_T)

    ct_inputs = [
        ct.TensorType(name="x",            shape=(1, d, T_dim, 1),          dtype=np.float16),
        ct.TensorType(name="rope_cos",      shape=(rope_T_dim, rope_half),   dtype=np.float16),
        ct.TensorType(name="rope_sin",      shape=(rope_T_dim, rope_half),   dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, T_dim, MAX_SEQ),   dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, MAX_SEQ, T_dim),   dtype=np.float16),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]

    n_shard = layer_end - layer_start
    ct_states = []
    for i in range(n_shard):
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, nkv, MAX_SEQ, dh), dtype=np.float16),
            name=f"layers.{i}.k_state"))
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, nkv, MAX_SEQ, dh), dtype=np.float16),
            name=f"layers.{i}.v_state"))

    print("  Converting to CoreML …")
    ml = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    print("  Quantizing weights INT8 …")
    ml = linear_quantize_weights(
        ml,
        config=OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8")))

    pkg_path = output_dir / f"{name}.mlpackage"
    ml.save(str(pkg_path))
    pkg_mb = sum(p.stat().st_size for p in pkg_path.rglob("*") if p.is_file()) / 1e6
    print(f"  Saved {pkg_path.name}  ({pkg_mb:.1f} MB)")

    mlmc = output_dir / f"{name}.mlmodelc"
    print(f"  Compiling → {mlmc.name} …")
    r = subprocess.run(
        ["xcrun", "coremlcompiler", "compile",
         str(pkg_path.resolve()), str(output_dir.resolve())],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  COMPILE ERROR:\n{r.stderr[:600]}")
        sys.exit(1)
    cml_mb = sum(p.stat().st_size for p in mlmc.rglob("*") if p.is_file()) / 1e6
    print(f"  Compiled  {mlmc.name}  ({cml_mb:.1f} MB)")
    return mlmc


def check_residency(mlmc: Path) -> bool:
    r = subprocess.run(
        [sys.executable,
         str(ROOT / "python" / "phi4_mini_residency_check.py"),
         str(mlmc)],
        capture_output=True, text=True)
    out = r.stdout + r.stderr
    passed = "PASS=True" in out and "conv_non_ane=0" in out and "compute_non_ane=0" in out
    tag = "PASS" if passed else "FAIL"
    summary = next((l for l in out.splitlines()
                    if any(k in l for k in ("conv_ane", "PASS", "FAIL", "compute_total"))), out[:300])
    print(f"  Residency: {tag}  —  {summary.strip()}")
    return passed


def make_rangedim_manifest(base_meta_path: Path, shard_mlmc_paths: list,
                           lm_head_shards_dir: Path, out_path: Path) -> None:
    with open(base_meta_path) as f:
        base = json.load(f)

    new_layers = []
    for (s, e), p in zip(SHARD_RANGES, shard_mlmc_paths):
        new_layers.append({"start": s, "end": e, "path": str(p.resolve())})

    # Look for T=4 LM head shards in the dedicated subdir
    # phi4_mini_lm_head_shards.py writes "lm_head_bt{N}_manifest.json"
    lm_head_manifest = next(lm_head_shards_dir.glob("lm_head_bt*_manifest.json"), None)
    lm_head_manifest = lm_head_manifest or (lm_head_shards_dir / "lm_head_manifest.json")
    if lm_head_manifest.exists():
        with open(lm_head_manifest) as f:
            lm_head_shards = json.load(f)
        print(f"  Using T=4 LM head shards from {lm_head_manifest}")
    else:
        # Fall back to existing T=1 shards from base manifest
        lm_head_shards = base.get("lm_head_shards", [])
        print(f"  WARNING: no T=4 LM head manifest found; using T=1 shards from base meta")

    out_meta = dict(base)
    out_meta["layers"] = new_layers
    out_meta["lm_head_shards"] = lm_head_shards
    out_meta.pop("speculative_verifier", None)
    out_meta["rangedim_t_max"] = T_MAX

    with open(out_path, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"\nManifest written → {out_path.name}")
    print(f"  layers: {[s['path'].split('/')[-1] for s in new_layers]}")


def main():
    parser = argparse.ArgumentParser(
        description="Build RangeDim (T=1..4) layer shards for HyMT 1.8B ANE")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end",   type=int, default=None)
    parser.add_argument("--skip-residency", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gguf = GGUFModel(str(GGUF_PATH))
    cfg  = gguf.config()
    print(f"HyMT: d={cfg['d_model']} nh={cfg['n_heads']} nkv={cfg['n_kv_heads']} "
          f"dh={cfg['d_head']} rope_dim={cfg.get('rope_dim', cfg['d_head'])} d_ff={cfg['d_ff']}")
    print(f"Trace T={TRACE_T}, RangeDim T∈[1..{T_MAX}], MAX_SEQ={MAX_SEQ}")

    ranges = SHARD_RANGES
    if args.layer_start is not None:
        ranges = [(args.layer_start, args.layer_end)]

    mlmc_paths = []
    for (s, e) in ranges:
        name = f"hymt_rangedim_layer{s}_{e}_q8"
        mlmc = export_shard(GGUF_PATH, cfg, s, e, OUTPUT_DIR, name)
        if not args.skip_residency:
            ok = check_residency(mlmc)
            if not ok:
                print(f"  WARNING: residency FAIL for shard [{s},{e})")
        mlmc_paths.append(mlmc)

    if len(mlmc_paths) == len(SHARD_RANGES) and args.layer_start is None:
        lm_head_dir = ROOT / "models" / "hymt" / "ane" / "lm_head_shards"
        make_rangedim_manifest(BASE_META, mlmc_paths, lm_head_dir, OUT_META)
        print("\n✓ All HyMT RangeDim shards built and manifest written.")
        print(f"  Manifest: {OUT_META.name}")
        print("  Next: build T=4 LM head shards:")
        print("    /usr/bin/python3 python/phi4_mini_lm_head_shards.py \\")
        print("      --model models/Hy-MT1.5-1.8B-2bit.gguf \\")
        print("      --output-dir models/hymt/ane/lm_head_shards \\")
        print("      --artifact-prefix HymtLMHead --num-shards 2 --batch-tokens 4 --force")
    else:
        print(f"\n✓ Built {len(mlmc_paths)} pilot shard(s).")
        print("  If residency PASS, re-run without --layer-start to build all shards.")


if __name__ == "__main__":
    main()
