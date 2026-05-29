#!/usr/bin/env python3
"""Build RangeDim (T=1..4) stateful layer shards for Phi-4-mini ANE.

A single compiled .mlmodelc handles both:
  - T=1 greedy decode (one real token, no dummy slots)
  - T=4 chunked prefill / block verification

Key differences from phi4_mini_t4_export_shard.py:
  - Traced at T=1 (greedy default)
  - KV scatter uses matmul instead of a Python for-loop (loop-free → T-agnostic)
  - CoreML inputs declared with ct.RangeDim(lower_bound=1, upper_bound=4, default=1)
    producing a single flexible compiled program (not multi-specialization).
  - No separate speculative_verifier manifest key: these shards ARE the production set

Source: Iverson APL inner-product principle (BOOK_ANALYSIS.md Exp 26) —
  single operator over the token axis replaces unrolled per-token loop.

Run with Xcode python3 (coremltools 9):
  /usr/bin/python3 python/phi4_mini_rangedim_export_shard.py

Optionally build one pilot shard first to validate residency before committing to all 4:
  /usr/bin/python3 python/phi4_mini_rangedim_export_shard.py --layer-start 30 --layer-end 32
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

GGUF_PATH   = ROOT / "models" / "phi4-mini" / "Phi-4-mini-instruct.Q8_0.gguf"
OUTPUT_DIR  = ROOT / "models" / "phi4-mini" / "ane" / "rangedim"
BASE_META   = ROOT / "models" / "phi4-mini" / \
              "phi4mini_runtime_meta.json"
OUT_META    = ROOT / "models" / "phi4-mini" / \
              "phi4mini_runtime_meta_rope96_rangedim_20_4_6_2.json"

TRACE_T     = 1       # trace at T=1 (greedy default)
T_MAX       = 4       # RangeDim upper bound; enables T=1..4 at runtime
MAX_SEQ     = 2048
QUANT_BITS  = 8

# Production shard topology (must match base manifest 20+4+6+2)
SHARD_RANGES = [(0, 20), (20, 24), (24, 30), (30, 32)]


# ─── Model definition ────────────────────────────────────────────────────────

def build_rangedim_shard(gguf, cfg, layer_start: int, layer_end: int) -> nn.Module:
    """Return a PyTorch T=TRACE_T stateful shard covering layers [layer_start, layer_end).

    The KV scatter uses a single matmul instead of a for-loop over T so the
    PyTorch graph contains no T-dependent control flow.  EnumeratedShapes then
    tells CoreML to compile T=1 and T=4 specializations from the same graph.
    """
    d         = cfg["d_model"]       # 3072
    nh        = cfg["n_heads"]       # 32
    nkv       = cfg["n_kv_heads"]    # 8
    dh        = cfg["d_head"]        # 96
    dff       = cfg["d_ff"]          # 8192
    eps       = cfg["rms_norm_eps"]
    rope_dim  = cfg.get("rope_dim", dh)  # 96
    rope_half = rope_dim // 2            # 48
    hpk       = nh // nkv                # 4 (query heads per KV group)
    kv_dim    = nkv * dh                 # 768
    qkv_dim   = d + 2 * kv_dim          # 4608

    class RMSNorm(nn.Module):
        def __init__(self, weight, eps_val):
            super().__init__()
            self.eps = eps_val
            self.w = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)
        def forward(self, x):              # x: [1, d, T, 1]
            K = x.shape[1] ** 0.5
            xs = x * (1.0 / K)
            v = xs.pow(2).mean(dim=1, keepdim=True)
            return (xs * torch.rsqrt(v + self.eps / (K * K)) * self.w).half()

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
            rope_cos:       [T, rope_half]   – per-position cosines (T rows)
            rope_sin:       [T, rope_half]
            attn_mask:      [1, 1, T, S]     – causal, -1e4=masked, 0=visible
            kv_write_mask:  [1, 1, S, T]     – col t has 1 at write position
            """
            residual = x
            nx = self.attn_norm(x)       # [1, d, T, 1]

            # ── QKV projection ──────────────────────────────────────────
            qkv = self.qkv(nx)           # [1, qkv_dim, T, 1]
            q_raw = qkv[:, :d, :, :]                    # [1, d, T, 1]
            k_raw = qkv[:, d:d + kv_dim, :, :]          # [1, kv_dim, T, 1]
            v_raw = qkv[:, d + kv_dim:, :, :]           # [1, kv_dim, T, 1]

            # ── RoPE (per-token positions) ───────────────────────────────
            # rope_cos/sin: [T, rope_half]
            # Traced at T=TRACE_T; EnumeratedShapes adapts the reshape dims.
            # Exp 26 layout fix: permute to [heads, dh, T] before final reshape
            # so memory layout matches channels-first [1, heads*dh, T, 1].
            def apply_rope(xc, n_heads):
                # xc: [1, n_heads*dh, T, 1]
                xc_sq = xc.squeeze(-1).permute(2, 0, 1)    # [T, 1, n_heads*dh]
                T_val = xc_sq.shape[0]
                xc_sq = xc_sq.reshape(T_val, n_heads, dh)  # [T, n_heads, dh]
                rot   = xc_sq[:, :, :rope_dim]             # [T, n_heads, rope_dim]
                pass_ = xc_sq[:, :, rope_dim:]
                lo    = rot[:, :, :rope_half]               # [T, n_heads, rope_half]
                hi    = rot[:, :, rope_half:]
                c = rope_cos.unsqueeze(1)                   # [T, 1, rope_half]
                s = rope_sin.unsqueeze(1)
                rot_out = torch.cat([lo * c - hi * s, lo * s + hi * c], dim=-1)
                out = torch.cat([rot_out, pass_], dim=-1)   # [T, n_heads, dh]
                # Permute to [n_heads, dh, T] then reshape → [1, n_heads*dh, T, 1]
                T_out = out.shape[0]
                out = out.permute(1, 2, 0).reshape(1, n_heads * dh, T_out, 1)
                return out

            q = apply_rope(q_raw, nh)   # [1, d, T, 1]
            k = apply_rope(k_raw, nkv)  # [1, kv_dim, T, 1]
            v = v_raw                   # [1, kv_dim, T, 1] — V has no RoPE

            # ── KV scatter: matmul replaces for-loop, handles any T natively ─
            # new_k: reshape [1, kv_dim, T, 1] → [1, nkv, T, dh]
            T_cur = v.shape[2]
            new_k = k.squeeze(-1).permute(0, 2, 1).reshape(1, T_cur, nkv, dh).permute(0, 2, 1, 3)
            new_v = v.squeeze(-1).permute(0, 2, 1).reshape(1, T_cur, nkv, dh).permute(0, 2, 1, 3)
            # new_k/new_v: [1, nkv, T, dh]

            # kv_write_mask: [1, 1, S, T]
            # torch.matmul([1, 1, S, T], [1, nkv, T, dh]) → [1, nkv, S, dh]
            # Works for any T — the T axis is a matmul inner dimension (dynamic).
            k_written = torch.matmul(kv_write_mask, new_k)   # [1, nkv, S, dh]
            v_written = torch.matmul(kv_write_mask, new_v)   # [1, nkv, S, dh]
            write_any = kv_write_mask.sum(dim=-1, keepdim=True)  # [1, 1, S, 1]

            k_updated = self.k_state * (1.0 - write_any) + k_written
            v_updated = self.v_state * (1.0 - write_any) + v_written
            self.k_state[:] = k_updated
            self.v_state[:] = v_updated

            # ── GQA attention ───────────────────────────────────────────
            # q: [1, d, T, 1] → [1, nh, T, dh]
            T_q = q.shape[2]
            q_h = q.squeeze(-1).permute(0, 2, 1).reshape(1, T_q, nh, dh).permute(0, 2, 1, 3)

            attn_parts = []
            scale = dh ** -0.5
            for kvi in range(nkv):
                q_g  = q_h[:, kvi * hpk:(kvi + 1) * hpk, :, :]  # [1, hpk, T, dh]
                k_h  = k_updated[:, kvi:kvi + 1, :, :]            # [1, 1, S, dh]
                v_h  = v_updated[:, kvi:kvi + 1, :, :]
                k_t  = k_h.transpose(-2, -1)                       # [1, 1, dh, S]
                sc   = torch.matmul(q_g, k_t) * scale             # [1, hpk, T, S]
                sc   = sc + attn_mask                              # [1, 1, T, S] broadcast
                aw   = torch.softmax(sc.float(), dim=-1).half()    # [1, hpk, T, S]
                ho   = torch.matmul(aw, v_h)                      # [1, hpk, T, dh]
                attn_parts.append(ho)

            # [1, nh, T, dh] → Exp 26 permute fix → [1, d, T, 1]
            ao = torch.cat(attn_parts, dim=1)                       # [1, nh, T, dh]
            T_ao = ao.shape[2]
            ao = ao.permute(0, 1, 3, 2).reshape(1, d, T_ao, 1)    # [1, d, T, 1]
            ao = self.out(ao)
            x = residual + ao

            # ── FFN ──────────────────────────────────────────────────────
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

    # Trace at T=TRACE_T
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

    # CoreML inputs — RangeDim for each T-dependent axis
    # Single flexible compiled program (not multi-specialization like EnumeratedShapes)
    T_dim       = ct.RangeDim(lower_bound=1, upper_bound=T_MAX, default=TRACE_T)
    rope_T_dim  = ct.RangeDim(lower_bound=1, upper_bound=T_MAX, default=TRACE_T)

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

    print("  Converting to CoreML (EnumeratedShapes) …")
    ml = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # INT8 weight quantization
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

    # Compile
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


# ─── Residency check (runs for each T in T_ENUM) ─────────────────────────────

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


# ─── Manifest generation ──────────────────────────────────────────────────────

def make_rangedim_manifest(base_meta_path: Path, shard_mlmc_paths: list,
                           out_path: Path) -> None:
    with open(base_meta_path) as f:
        base = json.load(f)

    # Replace the layers section with RangeDim shards
    new_layers = []
    for (s, e), p in zip(SHARD_RANGES, shard_mlmc_paths):
        new_layers.append({
            "start": s,
            "end":   e,
            "path":  str(p.resolve()),
        })

    # Reuse existing batch-4 LM head shards (already accept T=1..4 via EnumeratedShapes
    # or batch-4 fixed — they are separate shards and not affected by the T axis here)
    lm_head_shards = base.get("lm_head_shards", [])

    out_meta = dict(base)
    out_meta["layers"] = new_layers
    out_meta["lm_head_shards"] = lm_head_shards
    # Remove old speculative_verifier key if present (not needed — shards handle any T)
    out_meta.pop("speculative_verifier", None)
    out_meta["rangedim_t_max"] = T_MAX   # informational

    with open(out_path, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"\nManifest written → {out_path.name}")
    print(f"  layers: {[s['path'].split('/')[-1] for s in new_layers]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build RangeDim (T=1..4) layer shards for Phi-4-mini ANE")
    parser.add_argument("--layer-start", type=int, default=None,
                        help="Build only this shard range (pair with --layer-end)")
    parser.add_argument("--layer-end",   type=int, default=None)
    parser.add_argument("--skip-residency", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gguf = GGUFModel(str(GGUF_PATH))
    cfg  = gguf.config()
    print(f"Phi-4-mini: d={cfg['d_model']} nh={cfg['n_heads']} nkv={cfg['n_kv_heads']} "
          f"dh={cfg['d_head']} rope_dim={cfg.get('rope_dim', cfg['d_head'])}")
    print(f"Trace T={TRACE_T}, RangeDim T∈[1..{T_MAX}], MAX_SEQ={MAX_SEQ}")

    ranges = SHARD_RANGES
    if args.layer_start is not None:
        ranges = [(args.layer_start, args.layer_end)]

    mlmc_paths = []
    for (s, e) in ranges:
        name = f"phi4mini_rangedim_layer{s}_{e}_q8"
        mlmc = export_shard(GGUF_PATH, cfg, s, e, OUTPUT_DIR, name)
        if not args.skip_residency:
            ok = check_residency(mlmc)
            if not ok:
                print(f"  WARNING: residency FAIL for shard [{s},{e})")
                print("  Check whether RangeDim reshape propagation caused GPU/CPU fallback.")
        mlmc_paths.append(mlmc)

    if len(mlmc_paths) == len(SHARD_RANGES) and args.layer_start is None:
        make_rangedim_manifest(BASE_META, mlmc_paths, OUT_META)
        print("\n✓ All RangeDim shards built and manifest written.")
        print(f"  Use manifest: {OUT_META.name}")
        print("  Swift runtime update needed: pass T=1 for decode, T=4 for prefill/verification.")
    else:
        print(f"\n✓ Built {len(mlmc_paths)} pilot shard(s).")
        print("  If residency PASS, re-run without --layer-start to build all 4 shards.")


if __name__ == "__main__":
    main()
