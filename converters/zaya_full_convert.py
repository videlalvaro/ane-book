#!/usr/bin/env python3
"""ZAYA1-8B MoE shard exporter — Exp 34 (RangeDim T=1..4).

Builds 40 ANE-native MoE shards (layers 1,3,...,79) with:
  - RangeDim T=1..4 (greedy decode T=1 + chunked prefill T=4)
  - INT8 symmetric quantization (all conv/linear weights)
  - No MLState (MoE FFN is stateless per token)
  - Soft expert routing: compute all 16 expert FFNs, weight by router softmax
    This is T-agnostic: no dynamic control flow over routing decisions.

Architecture (ZAYA1-8B MoE layer):
  Router (T-agnostic MLP over router_dim=256):
    down_proj.weight     [256, H]     project hidden → router_dim
    down_proj.bias       [256]
    rmsnorm_eda.weight   [256]        RMSNorm on projected hidden
    router_mlp.0.weight  [256, 256]   MLP layer 0
    router_mlp.0.bias    [256]
    router_mlp.2.weight  [256, 256]   MLP layer 2
    router_mlp.2.bias    [256]
    router_mlp.4.weight  [17, 256]    Final projection → 17 logits (16 expert + 1 null)
    balancing_biases     [17]         load-balancing biases (applied at inference)

  16 Experts (SwiGLU, ffn_hidden_size=4096 = 2×2048):
    linear_fc1.weight    [4096, H]   gate+up fused (first 2048 = gate, last 2048 = up)
    linear_fc2.weight    [H, 2048]   down projection

  Residual gates (4 learned scale/bias):
    res_scale.hidden_states_scale [H]   scales MoE output
    res_scale.hidden_states_bias  [H]   bias on MoE output
    res_scale.residual_scale      [H]   scales residual input
    res_scale.residual_bias       [H]   bias on residual input

  Output: residual * residual_scale + residual_bias
        + moe_out * hidden_states_scale + hidden_states_bias

Soft routing formula (T-agnostic, no argmax dynamic control flow):
  router_weights = softmax(router_logits, dim=expert)   # [1, 17, T, 1]
  output = Σ_{e=0}^{15}  router_weights[:, e] * expert_e(normed_x)
  (Expert 16 = null slot, weight discarded)
  At inference with peaked routing, this is numerically equivalent to top-1
  hard routing. Numerical equivalence was validated in Exp 29 (T=1, PASS 9.3 tok/s).

Disk: ~7.7 GB compiled (40 shards × 193 MB each)
Compiler env: Xcode python3 (coremltools 9)

Run (gate-test one shard first):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
  local-artifacts/zaya_full_convert.py --gate-only

Run all 40 shards:
  TMPDIR=$PWD/local-artifacts/zaya_ane/cml_tmp \\
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
  local-artifacts/zaya_full_convert.py

Book refs:
  [Iverson APL §2] RangeDim as APL dynamic array semantics: one compiled
    program JIT-specialized at T=1..4 by E5RT.
  [Dragon Book §9.2] Soft routing = loop-invariant hoisting: the
    "compute all experts" approach eliminates T-dependent branch in the
    traced graph, enabling a single control-flow-free MIL program.
  [BOOK_ANALYSIS Exp32] Break-even analysis: with T=4 MoE, verifier cost
    drops from 475ms to ~130ms, break-even p>0.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# Script lives at local-artifacts/zaya_full_convert.py — go up 3 levels to repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
)

# ── Architecture constants ───────────────────────────────────────────────────

H           = 2048        # hidden_size
ROUTER_DIM  = 256         # zaya_mlp_expansion
N_EXPERTS   = 16          # num_experts
N_LOGITS    = 17          # router output slots (16 experts + 1 null)
FFN_HIDDEN  = 4096        # fc1 output = 2 × ffn_intermediate (SwiGLU fused)
FFN_INTER   = FFN_HIDDEN // 2   # 2048  (gate and up each)

NORM_EPS   = 1e-5
TRACE_T    = 1
T_MAX      = 4

MOE_LAYERS  = list(range(1, 80, 2))   # 1,3,5,...,79
SHARD_DIR   = ROOT / "emilio" / "conv-ane" / "zaya_ane" / "moe_rangedim"
META_IN     = ROOT / "emilio" / "conv-ane" / "zaya_ane" / "zaya_runtime_meta_stateful_cca.json"
META_OUT    = ROOT / "emilio" / "conv-ane" / "zaya_ane" / "zaya_runtime_meta_stateful_cca_rangedim.json"


# ── Path discovery ───────────────────────────────────────────────────────────

def find_weights_dir() -> Path:
    candidates = [
        ROOT / "models" / "ZAYA1-8B",
    ]
    # Also search any external volumes (e.g. an external SSD under <external-volume>/)
    import glob
    for vol in glob.glob("models/zaya/ZAYA1-8B"):
        candidates.append(Path(vol))
    for c in candidates:
        if (c / "model-00001-of-00004.safetensors").exists():
            return c
    for c in candidates:
        if c.exists():
            files = list(c.iterdir())
            print(f"  {c}: {[f.name for f in files[:8]]}")
    return None


# ── Safetensors weight loader ────────────────────────────────────────────────

def open_safetensors_for_layer(weights_dir: Path, layer_idx: int):
    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    key = f"model.layers.{layer_idx}.input_norm.weight"
    shard_filename = idx["weight_map"][key]
    shard_path = weights_dir / shard_filename
    from safetensors import safe_open
    sf = safe_open(str(shard_path), framework="pt", device="cpu")
    return sf, idx["weight_map"]


def load_moe_weights(weights_dir: Path, layer_idx: int) -> dict:
    """Load all MoE weights for one layer. Returns fp32 torch tensors.

    Because ZAYA safetensors shards may spread one layer across multiple
    physical .safetensors files, we open each key's shard individually.
    """
    p = f"model.layers.{layer_idx}"

    index_path = weights_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    wmap = idx["weight_map"]

    _sf_cache = {}
    from safetensors import safe_open

    def get(short_key):
        full_key = f"{p}.{short_key}"
        sf_file = wmap[full_key]
        if sf_file not in _sf_cache:
            _sf_cache[sf_file] = safe_open(
                str(weights_dir / sf_file), framework="pt", device="cpu")
        return _sf_cache[sf_file].get_tensor(full_key).to(torch.float32)

    w = {
        # Norms / residual gates
        "norm_w":           get("input_norm.weight"),
        "hs_scale":         get("res_scale.hidden_states_scale"),
        "hs_bias":          get("res_scale.hidden_states_bias"),
        "rs_scale":         get("res_scale.residual_scale"),
        "rs_bias":          get("res_scale.residual_bias"),
        # Router
        "router_down_w":    get("zaya_block.router.down_proj.weight"),   # [256, H]
        "router_down_b":    get("zaya_block.router.down_proj.bias"),     # [256]
        "router_norm_w":    get("zaya_block.router.rmsnorm_eda.weight"), # [256]
        "router_mlp0_w":    get("zaya_block.router.router_mlp.0.weight"),# [256,256]
        "router_mlp0_b":    get("zaya_block.router.router_mlp.0.bias"),  # [256]
        "router_mlp2_w":    get("zaya_block.router.router_mlp.2.weight"),# [256,256]
        "router_mlp2_b":    get("zaya_block.router.router_mlp.2.bias"),  # [256]
        "router_mlp4_w":    get("zaya_block.router.router_mlp.4.weight"),# [17,256]
        "router_biases":    get("zaya_block.router.balancing_biases"),   # [17]
    }
    # 16 expert weights
    for e in range(N_EXPERTS):
        w[f"fc1_{e}"] = get(f"zaya_block.experts.local_experts.{e}.linear_fc1.weight")  # [4096, H]
        w[f"fc2_{e}"] = get(f"zaya_block.experts.local_experts.{e}.linear_fc2.weight")  # [H, 2048]
    return w


# ── Inspection helper ────────────────────────────────────────────────────────

def inspect_shapes(weights_dir: Path):
    print("\n=== ZAYA1-8B Layer 1 MoE Tensor Shapes ===")
    w = load_moe_weights(weights_dir, 1)
    for k, v in w.items():
        print(f"  {k:<30s}  {tuple(v.shape)}  {v.dtype}")
    print(f"\n  Architecture summary:")
    print(f"  H={H}, ROUTER_DIM={ROUTER_DIM}, N_EXPERTS={N_EXPERTS}")
    print(f"  FFN_HIDDEN={FFN_HIDDEN} (gate={FFN_INTER}, up={FFN_INTER})")
    print(f"  TRACE_T={TRACE_T}, T_MAX={T_MAX}")


# ── PyTorch model — one ZAYA MoE layer ──────────────────────────────────────

def build_moe_shard(w: dict) -> nn.Module:
    """Return a traceable PyTorch module for one ZAYA MoE layer.

    The forward pass is T-agnostic:
      1. RMSNorm on input  [1, H, T, 1]
      2. Router MLP → softmax(logits)  [1, 17, T, 1]
      3. For each of 16 experts: SwiGLU FFN  [1, H, T, 1]
      4. Soft-weighted sum: Σ_e w_e * expert_e_out  [1, H, T, 1]
      5. Residual: x * rs_scale + rs_bias + moe * hs_scale + hs_bias

    All ops are Conv2d (1×1) or broadcast mul/add → no T-dependent control flow
    → RangeDim T=1..4 JIT-specialization by E5RT works correctly.

    Dragon Book §9.2 insight: "compute all experts" is the loop-invariant
    hoisting equivalent — the traced graph has no loop over routing decisions,
    only a fixed set of parallel Conv2d paths combined by broadcast multiply.
    """

    class RMSNorm(nn.Module):
        def __init__(self, w1d, dim):
            super().__init__()
            self.w = nn.Parameter(
                w1d.to(torch.float16).reshape(dim, 1, 1), requires_grad=False)
            self.eps = NORM_EPS
            self.dim = dim

        def forward(self, x):
            # x: [1, dim, T, 1]
            K = x.shape[1] ** 0.5
            xs = x * (1.0 / K)
            v = xs.pow(2).mean(dim=1, keepdim=True)
            return (xs * torch.rsqrt(v + self.eps / (K * K)) * self.w).half()

    class ZayaMoELayer(nn.Module):
        def __init__(self):
            super().__init__()

            # ── Input norm ──────────────────────────────────────────────
            self.input_norm = RMSNorm(w["norm_w"], H)

            # ── Residual gates ───────────────────────────────────────────
            self.register_buffer("hs_scale",
                w["hs_scale"].to(torch.float16).reshape(1, H, 1, 1))
            self.register_buffer("hs_bias",
                w["hs_bias"].to(torch.float16).reshape(1, H, 1, 1))
            self.register_buffer("rs_scale",
                w["rs_scale"].to(torch.float16).reshape(1, H, 1, 1))
            self.register_buffer("rs_bias",
                w["rs_bias"].to(torch.float16).reshape(1, H, 1, 1))

            # ── Router ──────────────────────────────────────────────────
            # down_proj: H → ROUTER_DIM, with bias
            self.router_down = nn.Conv2d(H, ROUTER_DIM, 1, bias=True)
            self.router_down.weight = nn.Parameter(
                w["router_down_w"].to(torch.float16).reshape(ROUTER_DIM, H, 1, 1),
                requires_grad=False)
            self.router_down.bias = nn.Parameter(
                w["router_down_b"].to(torch.float16), requires_grad=False)

            # rmsnorm_eda on router_dim
            self.router_norm = RMSNorm(w["router_norm_w"], ROUTER_DIM)

            # router_mlp.0: ROUTER_DIM → ROUTER_DIM, with bias
            self.router_mlp0 = nn.Conv2d(ROUTER_DIM, ROUTER_DIM, 1, bias=True)
            self.router_mlp0.weight = nn.Parameter(
                w["router_mlp0_w"].to(torch.float16).reshape(ROUTER_DIM, ROUTER_DIM, 1, 1),
                requires_grad=False)
            self.router_mlp0.bias = nn.Parameter(
                w["router_mlp0_b"].to(torch.float16), requires_grad=False)

            # router_mlp.2: ROUTER_DIM → ROUTER_DIM, with bias
            self.router_mlp2 = nn.Conv2d(ROUTER_DIM, ROUTER_DIM, 1, bias=True)
            self.router_mlp2.weight = nn.Parameter(
                w["router_mlp2_w"].to(torch.float16).reshape(ROUTER_DIM, ROUTER_DIM, 1, 1),
                requires_grad=False)
            self.router_mlp2.bias = nn.Parameter(
                w["router_mlp2_b"].to(torch.float16), requires_grad=False)

            # router_mlp.4: ROUTER_DIM → N_LOGITS (17), no bias
            self.router_mlp4 = nn.Conv2d(ROUTER_DIM, N_LOGITS, 1, bias=False)
            self.router_mlp4.weight = nn.Parameter(
                w["router_mlp4_w"].to(torch.float16).reshape(N_LOGITS, ROUTER_DIM, 1, 1),
                requires_grad=False)

            # Balancing biases added to router logits [17] at inference
            self.register_buffer("router_biases",
                w["router_biases"].to(torch.float16).reshape(1, N_LOGITS, 1, 1))

            # ── Expert FFNs (16 experts, SwiGLU) ────────────────────────
            # Each expert: fc1 (4096, H) → fc2 (H, 2048)
            # fc1 is SwiGLU: output [1, 4096, T, 1] split → gate[:2048] + up[2048:]
            self.fc1 = nn.ModuleList()
            self.fc2 = nn.ModuleList()
            for e in range(N_EXPERTS):
                c1 = nn.Conv2d(H, FFN_HIDDEN, 1, bias=False)
                c1.weight = nn.Parameter(
                    w[f"fc1_{e}"].to(torch.float16).reshape(FFN_HIDDEN, H, 1, 1),
                    requires_grad=False)
                self.fc1.append(c1)
                c2 = nn.Conv2d(FFN_INTER, H, 1, bias=False)
                c2.weight = nn.Parameter(
                    w[f"fc2_{e}"].to(torch.float16).reshape(H, FFN_INTER, 1, 1),
                    requires_grad=False)
                self.fc2.append(c2)

        def forward(self, x):
            """
            x: [1, H, T, 1]
            returns: [1, H, T, 1]
            """
            residual = x
            nx = self.input_norm(x)    # [1, H, T, 1]

            # ── Router MLP ──────────────────────────────────────────────
            r = self.router_down(nx)              # [1, 256, T, 1]
            r = F.silu(r)
            r = self.router_norm(r)               # [1, 256, T, 1]
            r = F.silu(self.router_mlp0(r))       # [1, 256, T, 1]
            r = F.silu(self.router_mlp2(r))       # [1, 256, T, 1]
            logits = self.router_mlp4(r)          # [1, 17, T, 1]
            logits = logits + self.router_biases  # [1, 17, T, 1]
            # Softmax over expert dim (dim=1) — router_weights sums to 1
            router_w = torch.softmax(logits.float(), dim=1).half()  # [1, 17, T, 1]

            # ── Expert FFNs (soft-weighted sum, T-agnostic) ──────────────
            # Dragon Book §9.2: loop-invariant hoisting.
            # Each expert FFN is a fixed Conv2d path — no T-dependent branch.
            # The routing decision is encoded in router_w[e], a scalar
            # broadcast over the token axis.
            moe_out = torch.zeros_like(nx)
            for e in range(N_EXPERTS):
                gate_up = self.fc1[e](nx)         # [1, 4096, T, 1]
                gate = gate_up[:, :FFN_INTER, :, :]   # [1, 2048, T, 1]
                up   = gate_up[:, FFN_INTER:, :, :]   # [1, 2048, T, 1]
                h = F.silu(gate) * up             # [1, 2048, T, 1]
                out_e = self.fc2[e](h)            # [1, H, T, 1]
                w_e = router_w[:, e:e+1, :, :]   # [1, 1, T, 1]
                moe_out = moe_out + out_e * w_e
            # Expert 16 (null slot) contributes zero — already handled by
            # the Σ only running over [0..15].

            # ── Residual with learned gating ─────────────────────────────
            return residual * self.rs_scale + self.rs_bias \
                 + moe_out  * self.hs_scale + self.hs_bias

    return ZayaMoELayer()


# ── Export one shard ─────────────────────────────────────────────────────────

def export_shard(weights_dir: Path, layer_idx: int, output_dir: Path,
                 skip_residency: bool = False) -> Path:
    label = f"zaya_moe_rangedim_L{layer_idx:02d}"
    print(f"\n{'='*60}")
    print(f"Building MoE RangeDim shard  L{layer_idx:02d}  →  {label}")
    print(f"  Trace T={TRACE_T}, RangeDim T∈[1..{T_MAX}]")
    print(f"{'='*60}")

    w = load_moe_weights(weights_dir, layer_idx)
    print(f"  Loaded {len(w)} weight tensors")
    print(f"    fc1_0: {tuple(w['fc1_0'].shape)}  fc2_0: {tuple(w['fc2_0'].shape)}")
    print(f"    router_mlp4: {tuple(w['router_mlp4_w'].shape)}")

    model = build_moe_shard(w)
    model.half().eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_buf    = sum(b.numel() for b in model.buffers())
    print(f"  params: {n_params:,}  buffers: {n_buf:,}")

    # ── Trace at T=1 ───────────────────────────────────────────────────
    T = TRACE_T
    x_ex = torch.randn(1, H, T, 1, dtype=torch.float16)
    with torch.no_grad():
        out_ex = model(x_ex)
        print(f"  trace output shape: {out_ex.shape}  dtype: {out_ex.dtype}")
    traced = torch.jit.trace(model, (x_ex,))

    # ── CoreML conversion ───────────────────────────────────────────────
    T_dim = ct.RangeDim(lower_bound=1, upper_bound=T_MAX, default=TRACE_T)
    ct_inputs  = [ct.TensorType(name="hidden", shape=(1, H, T_dim, 1), dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="moe_out", dtype=np.float16)]

    print("  Converting to CoreML …")
    ml = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    print("  Quantizing weights INT8 …")
    cfg = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    ml = linear_quantize_weights(ml, config=cfg)

    pkg_path = output_dir / f"{label}.mlpackage"
    ml.save(str(pkg_path))
    pkg_mb = sum(p.stat().st_size for p in pkg_path.rglob("*") if p.is_file()) / 1e6
    print(f"  Saved {pkg_path.name}  ({pkg_mb:.1f} MB)")

    mlmc = output_dir / f"{label}.mlmodelc"
    print(f"  Compiling → {mlmc.name} …")
    r = subprocess.run(
        ["xcrun", "coremlcompiler", "compile",
         str(pkg_path.resolve()), str(output_dir.resolve())],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  COMPILE ERROR:\n{r.stderr[:1200]}")
        sys.exit(1)
    cml_mb = sum(p.stat().st_size for p in mlmc.rglob("*") if p.is_file()) / 1e6
    print(f"  Compiled  {mlmc.name}  ({cml_mb:.1f} MB)")

    if not skip_residency:
        check_residency(mlmc)

    return mlmc


# ── ANE residency check ───────────────────────────────────────────────────────

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
                    if any(k in l for k in ("conv_ane", "PASS", "FAIL",
                                            "compute_total", "conv_non"))), out[:300])
    print(f"  Residency: {tag}  —  {summary.strip()}")
    return passed


# ── Manifest update ──────────────────────────────────────────────────────────

def update_manifest(mlmc_paths: dict, meta_in: Path, meta_out: Path):
    """Write new manifest replacing MoE shards with RangeDim versions."""
    with open(meta_in) as f:
        meta = json.load(f)

    new_layers = []
    for spec in meta["layers"]:
        layer_idx = spec["layer"]
        if spec["kind"] == "moe" and layer_idx in mlmc_paths:
            new_layers.append({
                "layer":    layer_idx,
                "kind":     "moe",
                "mlmodelc": str(mlmc_paths[layer_idx].resolve()),
            })
        else:
            new_layers.append(spec)

    meta["layers"] = new_layers
    meta["rangedim_t_max"] = T_MAX
    meta["moe_rangedim"] = True

    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nManifest written → {meta_out.name}")
    print(f"  RangeDim MoE layers: {sorted(mlmc_paths.keys())[:5]} ...")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ZAYA1-8B MoE RangeDim shard exporter (Exp 34)")
    ap.add_argument("--weights-dir", default=None,
                    help="Path to ZAYA1-8B safetensors weights directory")
    ap.add_argument("--output-dir", default=str(SHARD_DIR),
                    help=f"Output directory (default: {SHARD_DIR})")
    ap.add_argument("--gate-only", action="store_true",
                    help="Export only layer 1 (gate test) then stop")
    ap.add_argument("--layer", type=int, default=None,
                    help="Export a single specific MoE layer then stop")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip layers whose .mlmodelc already exists")
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--inspect", action="store_true",
                    help="Print layer 1 weight shapes and exit")
    ap.add_argument("--skip-residency", action="store_true",
                    help="Skip ANE residency check (faster, for bulk rebuild)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find weights ────────────────────────────────────────────────────
    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
    else:
        weights_dir = find_weights_dir()
    if weights_dir is None:
        print("ERROR: ZAYA1-8B weights not found. Checked:")
        print(f"  {ROOT / 'models' / 'ZAYA1-8B'}")
        print("  models/zaya/ZAYA1-8B")
        print("Provide --weights-dir or download weights to models/ZAYA1-8B.")
        sys.exit(1)
    print(f"Weights: {weights_dir}")

    # ── Inspect mode ────────────────────────────────────────────────────
    if args.inspect:
        inspect_shapes(weights_dir)
        return

    # ── Determine layers to build ────────────────────────────────────────
    if args.layer is not None:
        layers_to_build = [args.layer]
    elif args.gate_only:
        layers_to_build = [1]
    else:
        layers_to_build = MOE_LAYERS

    print(f"\nOutput dir: {output_dir}")
    print(f"Layers to build: {layers_to_build[:5]}{'...' if len(layers_to_build) > 5 else ''}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Skip residency: {args.skip_residency}")
    print()

    mlmc_paths = {}
    failed = []

    for layer_idx in layers_to_build:
        label = f"zaya_moe_rangedim_L{layer_idx:02d}"
        mlmc = output_dir / f"{label}.mlmodelc"

        if args.skip_existing and mlmc.exists():
            cml_mb = sum(p.stat().st_size for p in mlmc.rglob("*") if p.is_file()) / 1e6
            print(f"  SKIP L{layer_idx:02d} (already exists, {cml_mb:.1f} MB)")
            mlmc_paths[layer_idx] = mlmc
            continue

        try:
            mlmc = export_shard(weights_dir, layer_idx, output_dir,
                                skip_residency=args.skip_residency)
            mlmc_paths[layer_idx] = mlmc
        except Exception as e:
            print(f"  ERROR on L{layer_idx:02d}: {e}")
            failed.append(layer_idx)
            if args.gate_only:
                raise

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. Built {len(mlmc_paths)}/{len(layers_to_build)} shards.")
    if failed:
        print(f"FAILED layers: {failed}")

    # ── Update manifest (only if running full set or gate+success) ───────
    if not args.gate_only and len(mlmc_paths) > 0 and not failed:
        if META_IN.exists():
            update_manifest(mlmc_paths, META_IN, META_OUT)
            print(f"\nNext: run zaya_ane_runtime with --meta {META_OUT.name}")
        else:
            print(f"\nWARN: meta_in not found: {META_IN}")
            print("Manifest not updated.")

    elif args.gate_only and mlmc_paths:
        lyr = list(mlmc_paths.keys())[0]
        print(f"\nGate test complete. Shard: {mlmc_paths[lyr]}")
        print(f"Run full build: python3 {__file__}")


if __name__ == "__main__":
    main()
