#!/usr/bin/env python3
"""LFM2.5-8B-A1B ANE Converter — LiquidAI ShortConv + MoE on Apple Neural Engine.

Converts LiquidAI/LFM2.5-8B-A1B from HuggingFace safetensors to CoreML
INT8 mlpackage shards, all targeting the Apple Neural Engine.

Architecture (24 layers total):
  Layer type per index (from config.json layer_types):
    conv layers (18): Lfm2MoeShortConv + MoE FFN
    full_attention (6): GQA Attention + MoE FFN
    layers 0,1 (both conv): use dense MLP instead of MoE (num_dense_layers=2)

  Lfm2MoeShortConv (the novel operator) — "double-gated LIV conv":
    BCx = in_proj(normed_x)               # Linear(H → 3H)
    B, C, x = BCx.chunk(3)
    Bx = B * x                            # first gate
    conv_out = causal_depthwise_conv1d(Bx, kernel=3, cache=3)  # tiny state!
    y = C * conv_out                      # second gate
    output = out_proj(y)                  # Linear(H → H)
    State = [1, H, 3, 1] per layer = 6144 floats (vs O(T) for KV cache)

  MoE FFN (22 layers: layers 2-23):
    router:  Linear(H → 32) → sigmoid → top-4 select (soft-routed for ANE)
    experts: 32 × SiLU-gate FFN(H → 1792 → H), split into 2 shards of 16
    Per-half shard: ~177MB INT8 (under 250MB ANE ceiling)

ANE mapping:
  - all Linear → Conv2d(C_in, C_out, 1, 1)
  - depthwise Conv1d(H, H, kernel=3) → Conv2d(H, H, kernel=(3,1), groups=H)
  - conv state sliding window: cat([state[:,1:,:], Bx], dim=2) — pure tensor op
  - soft MoE: 16 expert chains unrolled at trace time, summed — no branches

Shard types per layer:
  1. operator shard  : operator_norm + ShortConv|GQA + residual + ffn_norm + router
                       Output: updated_hidden, ffn_normed, routing_weights[32]
  2. moe_A shard     : 16 experts (0–15), soft-routed → contribution_A
  3. moe_B shard     : 16 experts (16–31), soft-routed → contribution_B
  4. dense MLP shard : (layers 0,1 only) dense SiLU-gate FFN → ffn_out
  Host per layer:
     hidden = updated_hidden + contribution_A + contribution_B

LM head: tied to embed_tokens (vocab=128000 × 2048 = 262MB at INT8)
  → 2 vocab-half shards of 64000 × 2048 = ~131MB each

Disk estimate: ~72 shards × avg ~90MB = ~6.5 GB compiled INT8

Requirements:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3  (coremltools 9)
  safetensors, torch, numpy

Run:
    TMPDIR=$PWD/ane-book/lfm25_ane/cml_tmp \\
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
  converters/lfm25_convert.py \\
    --weights LiquidAI/LFM2.5-8B-A1B/model.safetensors \\
    --out-dir models/lfm25/ane \\
    --gate-only    # converts layer 0 only for ANE residency gate

Gate test (run first!):
  python3 validators/lfm25_residency_check.py --shard models/lfm25/ane/lfm25_op_layer0.mlmodelc

Book refs:
  [Dragon Book §9.2] Soft MoE routing = loop-invariant hoisting: the loop
    over 16 experts is unrolled at trace time → single branch-free MIL program.
  [Iverson APL §2]  ShortConv state update = array shift: cat([s[:,1:,:], Bx])
    is a rank-preserving APL-style rotate — maps to two tensor ops on ANE.
  [Knuth Vol 3 §6.4] Conv state as a circular buffer of depth L=3:
    optimal for cache-resident streaming state updates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model config (from config.json, hardcoded for reproducibility)
# ---------------------------------------------------------------------------

LAYER_TYPES = [
    "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention",
    "conv", "conv", "full_attention",
    "conv", "conv",
]  # 24 entries: 18 conv + 6 full_attention

HIDDEN_SIZE          = 2048
INTERMEDIATE_SIZE    = 7168   # dense MLP (layers 0,1)
MOE_INTERMEDIATE     = 1792   # per-expert FFN dim
NUM_EXPERTS          = 32
NUM_EXPERTS_PER_TOK  = 4
NUM_HIDDEN_LAYERS    = 24
NUM_DENSE_LAYERS     = 2      # first N layers use dense MLP
CONV_L_CACHE         = 3      # short-conv state depth
NUM_ATTN_HEADS       = 32
NUM_KV_HEADS         = 8
HEAD_DIM             = HIDDEN_SIZE // NUM_ATTN_HEADS   # 64
VOCAB_SIZE           = 128_000
NORM_EPS             = 1e-5
ROPE_THETA           = 5_000_000.0
NORM_TOPK_PROB       = True
USE_EXPERT_BIAS      = True
ROUTED_SCALING       = 1.0

MOE_EXPERTS_PER_HALF = NUM_EXPERTS // 2   # 16 experts per shard
MAX_SEQ             = 2048              # pre-allocated KV cache depth for attention shards

# ---------------------------------------------------------------------------
# ANE-compatible PyTorch modules
# ---------------------------------------------------------------------------

class ANERMSNorm(nn.Module):
    """RMSNorm with ANE-friendly per-channel implementation."""
    def __init__(self, hidden_size: int, eps: float = NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.H = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, H, 1, 1]
        # var across channel dim, rsqrt, scale — all ANE-native elementwise
        variance = (x * x).mean(dim=1, keepdim=True)                # [1, 1, 1, 1]
        x_normed = x * torch.rsqrt(variance + self.eps)              # [1, H, 1, 1]
        return x_normed * self.weight.view(1, self.H, 1, 1)


class ANEShortConvDecode(nn.Module):
    """ShortConv decode shard (T=1) — the core LFM2.5 novel operator.

    Maps to ANE via:
      in_proj   → Conv2d(H, 3H, 1×1)           : 1×1 conv = linear projection
      first gate → elementwise B * x            : ANE-native
      state update → cat([state[:,1:,:], Bx])   : slice + concat, no scan
      conv_decode → Conv2d(H, H, (L,1), groups=H): depthwise kernel over L positions
      second gate → elementwise C * conv_out    : ANE-native
      out_proj  → Conv2d(H, H, 1×1)            : 1×1 conv

    State shape: [1, H, L, 1] — only 3 × 2048 floats per layer.
    No MLState needed: state is passed as a regular input/output.
    (KV cache needs MLState because it grows with T; conv state is always size L.)
    """
    def __init__(self, hidden_size: int, L_cache: int):
        super().__init__()
        H, L = hidden_size, L_cache
        self.H, self.L = H, L
        self.in_proj     = nn.Conv2d(H, 3 * H, 1, bias=False)
        # Depthwise kernel (L, 1) over the L-position sliding window
        # Input: [1, H, L, 1] → output: [1, H, 1, 1] (valid conv, no padding)
        self.conv_decode = nn.Conv2d(H, H, (L, 1), groups=H, bias=False)
        self.out_proj    = nn.Conv2d(H, H,     1, bias=False)

    def forward(self, hidden: torch.Tensor, conv_state: torch.Tensor):
        # hidden:     [1, H, 1, 1]
        # conv_state: [1, H, L, 1]  (positions 0=oldest .. L-1=newest)
        BCx = self.in_proj(hidden)                       # [1, 3H, 1, 1]
        B   = BCx[:, :self.H,           :, :]
        C   = BCx[:, self.H:2*self.H,   :, :]
        x   = BCx[:, 2*self.H:,         :, :]
        Bx  = B * x                                      # first gate: [1, H, 1, 1]
        # Slide window: drop position 0, append Bx at end
        new_state = torch.cat([conv_state[:, :, 1:, :], Bx], dim=2)  # [1, H, L, 1]
        # Apply depthwise kernel over the full L-window
        conv_out = self.conv_decode(new_state)            # [1, H, 1, 1]
        y = C * conv_out                                  # second gate
        return self.out_proj(y), new_state


class ANEDenseMLP(nn.Module):
    """Dense SiLU-gate FFN (layers 0, 1 only). intermediate_size=7168."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Conv2d(hidden_size, intermediate_size, 1, bias=False)
        self.w3 = nn.Conv2d(hidden_size, intermediate_size, 1, bias=False)
        self.w2 = nn.Conv2d(intermediate_size, hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) * up → down
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ANEMoEHalf(nn.Module):
    """16 soft-routed ANE experts.

    All 16 experts run during tracing → CoreML gets 16 independent
    SiLU-gate FFN chains with no branches. Their outputs are weighted
    by the router sigmoid scores and summed.

    [Dragon Book §9.2]: loop-invariant code motion applied at trace time;
    each expert path is fully static and branch-free in the MIL program.
    """
    def __init__(self, hidden_size: int, moe_intermediate_size: int, n_half: int = MOE_EXPERTS_PER_HALF):
        super().__init__()
        H, D, N = hidden_size, moe_intermediate_size, n_half
        self.H, self.D, self.N = H, D, N
        for i in range(N):
            setattr(self, f"gate_proj_{i}", nn.Conv2d(H, D, 1, bias=False))
            setattr(self, f"up_proj_{i}",   nn.Conv2d(H, D, 1, bias=False))
            setattr(self, f"down_proj_{i}", nn.Conv2d(D, H, 1, bias=False))

    def forward(self, ffn_normed: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        # ffn_normed:      [1, H, 1, 1]
        # routing_weights: [1, N, 1, 1]  (sigmoid scores for these N experts)
        contributions = []
        for i in range(self.N):
            w_i    = routing_weights[:, i : i + 1, :, :]        # [1, 1, 1, 1]
            gate_i = F.silu(getattr(self, f"gate_proj_{i}")(ffn_normed))
            up_i   = getattr(self, f"up_proj_{i}")(ffn_normed)
            down_i = getattr(self, f"down_proj_{i}")(gate_i * up_i)
            contributions.append(w_i * down_i)
        # sum over all N expert contributions — fully unrolled, no control flow
        return sum(contributions)


class ANEConvOperatorShard(nn.Module):
    """Full decode pass for one conv-type layer.

    Computes:
      normed = operator_norm(hidden)
      (conv_out, new_state) = ShortConv(normed, conv_state)
      hidden = hidden + conv_out
      ffn_normed = ffn_norm(hidden)
      routing_weights = sigmoid(router(ffn_normed))

    Returns: updated_hidden, new_conv_state, ffn_normed, routing_weights
    """
    def __init__(self, hidden_size: int, L_cache: int, n_experts: int):
        super().__init__()
        H = hidden_size
        self.operator_norm = ANERMSNorm(H)
        self.short_conv     = ANEShortConvDecode(H, L_cache)
        self.ffn_norm       = ANERMSNorm(H)
        self.router         = nn.Conv2d(H, n_experts, 1, bias=False)

    def forward(self, hidden: torch.Tensor, conv_state: torch.Tensor):
        normed = self.operator_norm(hidden)
        conv_out, new_state = self.short_conv(normed, conv_state)
        updated_hidden = hidden + conv_out
        ffn_normed = self.ffn_norm(updated_hidden)
        routing_weights = torch.sigmoid(self.router(ffn_normed))    # [1, 32, 1, 1]
        return updated_hidden, new_state, ffn_normed, routing_weights


class ANEAttnOperatorShard(nn.Module):
    """Full decode pass for one full_attention-type layer (GQA, T=1).

    Note: KV cache is NOT included here — KV state management for the
    6 attention layers uses a stateful approach in the Swift runtime
    (same pattern as Phi-4 mini). This shard handles projections + RoPE.

    For the full production version with MLState KV cache, see the
    stateful variant in runtime/lfm25_ane.swift.
    """
    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int, n_experts: int):
        super().__init__()
        H = hidden_size
        Q_DIM = n_heads    * head_dim   # 2048
        KV_DIM = n_kv_heads * head_dim  # 512
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.n_kv_groups = n_heads // n_kv_heads   # 4

        self.operator_norm = ANERMSNorm(H)
        self.q_proj        = nn.Conv2d(H, Q_DIM,  1, bias=False)
        self.k_proj        = nn.Conv2d(H, KV_DIM, 1, bias=False)
        self.v_proj        = nn.Conv2d(H, KV_DIM, 1, bias=False)
        self.q_layernorm   = ANERMSNorm(head_dim)
        self.k_layernorm   = ANERMSNorm(head_dim)
        self.out_proj      = nn.Conv2d(Q_DIM, H, 1, bias=False)
        self.ffn_norm      = ANERMSNorm(H)
        self.router        = nn.Conv2d(H, n_experts, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,    # [1, H, 1, 1]
        k_cache: torch.Tensor,   # [1, KV_DIM, T_past, 1]
        v_cache: torch.Tensor,   # [1, KV_DIM, T_past, 1]
        cos: torch.Tensor,       # [1, head_dim, 1, 1]
        sin: torch.Tensor,       # [1, head_dim, 1, 1]
    ):
        H = self.n_heads * self.head_dim
        normed = self.operator_norm(hidden)                         # [1, H, 1, 1]

        # Project: [1, H, 1, 1] → [1, Q/K/V_DIM, 1, 1]
        q = self.q_proj(normed)   # [1, n_heads*head_dim, 1, 1]
        k = self.k_proj(normed)   # [1, n_kv_heads*head_dim, 1, 1]
        v = self.v_proj(normed)   # [1, n_kv_heads*head_dim, 1, 1]

        # Per-head layernorm on q and k (LFM2 uses QK-norm)
        # Reshape to per-head for norm: [1, head_dim, n_heads, 1] → norm → back
        q_h = q.reshape(1, self.head_dim, self.n_heads, 1)
        k_h = k.reshape(1, self.head_dim, self.n_kv_heads, 1)
        q_h = self.q_layernorm(q_h)                                # per-head RMSNorm
        k_h = self.k_layernorm(k_h)
        # Apply RoPE: elementwise cos + rotate_half * sin
        q_rot = q_h * cos - q_h.flip(1) * sin   # simplified — host precomputes cos/sin
        k_rot = k_h * cos - k_h.flip(1) * sin

        q_rot = q_rot.reshape(1, H, 1, 1)
        k_rot = k_rot.reshape(1, self.n_kv_heads * self.head_dim, 1, 1)

        # Append to KV cache (host manages cache — shard outputs new k, v)
        new_k = torch.cat([k_cache, k_rot], dim=2)                # [1, KV_DIM, T+1, 1]
        new_v = torch.cat([v_cache, v_rot := v], dim=2)           # [1, KV_DIM, T+1, 1]

        # GQA: expand KV to match query heads
        # For ANE: repeat KV along head dim to match n_heads
        # k_exp: [1, n_heads*head_dim, T+1, 1]
        k_exp = new_k.reshape(1, self.n_kv_heads, self.head_dim, -1)
        k_exp = k_exp.repeat(1, self.n_kv_groups, 1, 1)
        k_exp = k_exp.reshape(1, H, -1, 1)

        v_exp = new_v.reshape(1, self.n_kv_heads, self.head_dim, -1)
        v_exp = v_exp.repeat(1, self.n_kv_groups, 1, 1)
        v_exp = v_exp.reshape(1, H, -1, 1)

        # Scaled dot-product attention (T_past+1 positions)
        # [1, n_heads, 1, head_dim] × [1, n_heads, head_dim, T+1] → [1, n_heads, 1, T+1]
        T_new = new_k.shape[2]
        q_for_attn = q_rot.reshape(1, self.n_heads, self.head_dim, 1)
        k_for_attn = k_exp.reshape(1, self.n_heads, self.head_dim, T_new)
        v_for_attn = v_exp.reshape(1, self.n_heads, self.head_dim, T_new)

        scale = self.head_dim ** -0.5
        attn_logits = torch.einsum("bhdq,bhdT->bhqT", q_for_attn, k_for_attn) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)              # [1, n_heads, 1, T+1]
        attn_out = torch.einsum("bhqT,bhdT->bhdq", attn_weights, v_for_attn)
        attn_out = attn_out.reshape(1, H, 1, 1)

        attn_result = self.out_proj(attn_out)
        updated_hidden = hidden + attn_result
        ffn_normed = self.ffn_norm(updated_hidden)
        routing_weights = torch.sigmoid(self.router(ffn_normed))   # [1, 32, 1, 1]
        return updated_hidden, new_k, new_v, ffn_normed, routing_weights


class ANEAttnDecodeFixed(nn.Module):
    """GQA decode shard with fixed-size KV cache (all shapes static → ANE-only).

    KV cache is pre-allocated to MAX_SEQ tokens and updated via one-hot write_mask
    scatter (elementwise only — no dynamic reshape). GQA is unrolled over n_kv_heads=8
    at trace time → 8 static matmul chains, zero branches in the MIL program.

    Inputs:
      hidden      [1, H,      1,       1]
      k_cache     [1, KV_DIM, MAX_SEQ, 1]  pre-allocated, all zeros at step 0
      v_cache     [1, KV_DIM, MAX_SEQ, 1]
      write_mask  [1, 1,      MAX_SEQ, 1]  one-hot: 1.0 at current position
      attn_mask   [1, 1,      1,       MAX_SEQ]  0=attend, -1e4=future/padding
      cos         [1, HEAD_DIM, 1,     1]  RoPE cosines for current position
      sin         [1, HEAD_DIM, 1,     1]  RoPE sines for current position

    Outputs: updated_hidden, new_k, new_v, ffn_normed, routing_weights

    Book refs:
      [Dragon Book §9.2] GQA loop unrolled at trace time: 8 kv-head chains
        compiled into a single branch-free MIL program (same principle as MoE).
      [Knuth Vol 3 §5.2] Fixed-size pre-allocated cache with one-hot scatter
        = Knuth's address-calculation insertion into a static table.
    """
    def __init__(
        self,
        hidden_size:  int,
        n_heads:      int,
        n_kv_heads:   int,
        head_dim:     int,
        n_experts:    int,
        max_seq:      int,
    ):
        super().__init__()
        H      = hidden_size
        Q_DIM  = n_heads   * head_dim   # 2048
        KV_DIM = n_kv_heads * head_dim  # 512
        self.n_heads     = n_heads      # 32
        self.n_kv_heads  = n_kv_heads   # 8
        self.head_dim    = head_dim     # 64
        self.n_kv_groups = n_heads // n_kv_heads  # 4
        self.max_seq     = max_seq

        self.operator_norm = ANERMSNorm(H)
        self.q_proj        = nn.Conv2d(H,      Q_DIM,     1, bias=False)
        self.k_proj        = nn.Conv2d(H,      KV_DIM,    1, bias=False)
        self.v_proj        = nn.Conv2d(H,      KV_DIM,    1, bias=False)
        self.q_layernorm   = ANERMSNorm(head_dim)
        self.k_layernorm   = ANERMSNorm(head_dim)
        self.out_proj      = nn.Conv2d(Q_DIM,  H,         1, bias=False)
        self.ffn_norm      = ANERMSNorm(H)
        self.router        = nn.Conv2d(H,      n_experts, 1, bias=False)

    def forward(
        self,
        hidden:     torch.Tensor,   # [1, H,       1,       1]
        k_cache:    torch.Tensor,   # [1, KV_DIM,  MAX_SEQ, 1]
        v_cache:    torch.Tensor,   # [1, KV_DIM,  MAX_SEQ, 1]
        write_mask: torch.Tensor,   # [1, 1,       MAX_SEQ, 1]  one-hot
        attn_mask:  torch.Tensor,   # [1, 1,       1,       MAX_SEQ]
        cos:        torch.Tensor,   # [1, head_dim, 1,      1]
        sin:        torch.Tensor,   # [1, head_dim, 1,      1]
    ):
        H      = self.n_heads    * self.head_dim   # 2048
        KV_DIM = self.n_kv_heads * self.head_dim   # 512
        dh     = self.head_dim                      # 64
        dh2    = dh // 2                            # 32

        normed = self.operator_norm(hidden)    # [1, H, 1, 1]

        # QKV projections
        q = self.q_proj(normed)   # [1, H,      1, 1]
        k = self.k_proj(normed)   # [1, KV_DIM, 1, 1]
        v = self.v_proj(normed)   # [1, KV_DIM, 1, 1]

        # Per-head QK-norm (LFM2 applies RMSNorm per head before RoPE)
        # q_proj output is head-major: q[h*dh + d] = head h, dim d
        # Reshape to [1, dh, n_heads, 1] for per-head norm — requires permute to
        # get correct head grouping (otherwise dim=1 normalizes across heads, not per-head).
        q_h = q.reshape(1, self.n_heads,    dh, 1).permute(0, 2, 1, 3)  # [1, 64, 32, 1]
        k_h = k.reshape(1, self.n_kv_heads, dh, 1).permute(0, 2, 1, 3)  # [1, 64,  8, 1]
        q_h = self.q_layernorm(q_h)
        k_h = self.k_layernorm(k_h)

        # RoPE — standard rotate_half; cos/sin [1, head_dim, 1, 1] broadcast over heads
        q1, q2 = q_h[:, :dh2, :, :], q_h[:, dh2:, :, :]
        k1, k2 = k_h[:, :dh2, :, :], k_h[:, dh2:, :, :]
        q_rot = q_h * cos + torch.cat([-q2, q1], dim=1) * sin   # [1, 64, 32, 1]
        k_rot = k_h * cos + torch.cat([-k2, k1], dim=1) * sin   # [1, 64,  8, 1]
        # Convert back to head-major for KV cache storage and GQA attention loop
        q_rot = q_rot.permute(0, 2, 1, 3).reshape(1, H,      1, 1)   # [1, 2048, 1, 1] head-major
        k_rot = k_rot.permute(0, 2, 1, 3).reshape(1, KV_DIM, 1, 1)   # [1,  512, 1, 1] head-major

        # KV cache scatter-write: one-hot write_mask broadcasts over KV_DIM
        # All shapes fixed → no RangeDim needed, purely static ANE graph
        new_k = k_cache * (1 - write_mask) + k_rot * write_mask  # [1, KV_DIM, MAX_SEQ, 1]
        new_v = v_cache * (1 - write_mask) + v     * write_mask  # [1, KV_DIM, MAX_SEQ, 1]

        # GQA decode attention — loop unrolled over n_kv_heads=8 at trace time
        # Each iteration: 1 key head × n_kv_groups=4 query heads  [Dragon Book §9.2]
        scale = dh ** -0.5
        attn_chunks = []
        for kv_i in range(self.n_kv_heads):
            q_s = kv_i * self.n_kv_groups * dh
            k_s = kv_i * dh
            k_e = k_s + dh

            # q group: [1, n_kv_groups, 1, dh]
            q_g = q_rot[:, q_s : q_s + self.n_kv_groups * dh, :, :]
            q_g = q_g.reshape(1, self.n_kv_groups, 1, dh)       # [1, 4, 1, 64]

            # k/v slice for this head: [1, dh, MAX_SEQ, 1]
            k_hi = new_k[:, k_s:k_e, :, :].reshape(1, 1, dh, self.max_seq)  # [1, 1, 64, S]
            v_hi = new_v[:, k_s:k_e, :, :].permute(0, 3, 2, 1)             # [1, 1, S, 64]

            # Attention scores: [1,4,1,64] × [1,1,64,S] → [1,4,1,S]
            scores  = torch.matmul(q_g, k_hi) * scale
            scores  = scores + attn_mask                    # causal mask broadcast
            weights = F.softmax(scores, dim=-1)             # [1, 4, 1, S]

            # Weighted value: [1,4,1,S] × [1,1,S,64] → [1,4,1,64]
            out_i = torch.matmul(weights, v_hi)
            out_i = out_i.reshape(1, self.n_kv_groups * dh, 1, 1)
            attn_chunks.append(out_i)

        attn_out = torch.cat(attn_chunks, dim=1)   # [1, H, 1, 1]
        attn_result    = self.out_proj(attn_out)
        updated_hidden = hidden + attn_result

        ffn_normed      = self.ffn_norm(updated_hidden)
        routing_weights = torch.sigmoid(self.router(ffn_normed))  # [1, 32, 1, 1]
        return updated_hidden, new_k, new_v, ffn_normed, routing_weights


class ANEDenseLayerShard(nn.Module):
    """Layers 0, 1: conv operator + dense MLP (no MoE). All-in-one shard."""
    def __init__(self, hidden_size: int, intermediate_size: int, L_cache: int):
        super().__init__()
        H = hidden_size
        self.operator_norm = ANERMSNorm(H)
        self.short_conv     = ANEShortConvDecode(H, L_cache)
        self.ffn_norm       = ANERMSNorm(H)
        self.dense_ffn      = ANEDenseMLP(H, intermediate_size)

    def forward(self, hidden: torch.Tensor, conv_state: torch.Tensor):
        # Conv operator path
        normed = self.operator_norm(hidden)
        conv_out, new_state = self.short_conv(normed, conv_state)
        updated_hidden = hidden + conv_out
        # Dense FFN path
        ffn_normed = self.ffn_norm(updated_hidden)
        ffn_out = self.dense_ffn(ffn_normed)
        final_hidden = updated_hidden + ffn_out
        return final_hidden, new_state


class ANELMHeadHalf(nn.Module):
    """Half of the LM head (64K vocab slice) for vocab-split ANE sharding.

    lm_head.weight is tied to embed_tokens.weight (128000 × 2048).
    At INT8: 262MB → split into 2 × 131MB shards.
    """
    def __init__(self, hidden_size: int, vocab_half: int):
        super().__init__()
        self.proj = nn.Conv2d(hidden_size, vocab_half, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [1, H, 1, 1] → [1, vocab_half, 1, 1]
        return self.proj(hidden)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def load_weights_for_layer(st, layer_idx: int, verbose: bool = False):
    """Lazily load all weights for one layer from safetensors.

    Returns a dict of {suffix: tensor} for the layer.
    Tensors are float32 regardless of storage dtype (BF16 in checkpoint).
    """
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in st.keys() if k.startswith(prefix)]
    if verbose:
        print(f"  Layer {layer_idx}: {len(keys)} tensors")
    return {
        k[len(prefix):]: st.get_tensor(k).to(dtype=torch.float32)
        for k in keys
    }


def load_global_weights(st) -> dict:
    """Load embedding, final norm, lm_head weights.

    lm_head.weight is NOT a separate key in the checkpoint: it is tied to
    model.embed_tokens.weight (tie_word_embeddings=True in config.json).
    The HF serializer omits the duplicate; we reuse the same tensor.
    """
    embed = st.get_tensor("model.embed_tokens.weight").to(torch.float32)
    return {
        "embed_tokens": embed,
        "embedding_norm": st.get_tensor("model.embedding_norm.weight").to(torch.float32),
        "lm_head": embed,  # tied — same tensor, no separate checkpoint key
    }


# ---------------------------------------------------------------------------
# Module weight-loading functions
# ---------------------------------------------------------------------------

def populate_conv_operator_shard(module: ANEConvOperatorShard, w: dict, layer_idx: int):
    """Fill ANEConvOperatorShard with weights from the layer weight dict."""
    assert LAYER_TYPES[layer_idx] == "conv", f"Layer {layer_idx} is not a conv layer"

    # Norms
    module.operator_norm.weight.data = w["operator_norm.weight"].view(HIDDEN_SIZE)
    module.ffn_norm.weight.data       = w["ffn_norm.weight"].view(HIDDEN_SIZE)

    # ShortConv
    sc = module.short_conv
    # in_proj weight: [6144, 2048] Linear → Conv2d [6144, 2048, 1, 1]
    sc.in_proj.weight.data    = w["conv.in_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    # conv weight: [H, 1, L] depthwise Conv1d → Conv2d [H, 1, L, 1]
    sc.conv_decode.weight.data = w["conv.conv.weight"].unsqueeze(-1)  # [H, 1, L] → [H, 1, L, 1]
    # out_proj weight: [2048, 2048] Linear → Conv2d [2048, 2048, 1, 1]
    sc.out_proj.weight.data   = w["conv.out_proj.weight"].unsqueeze(-1).unsqueeze(-1)

    # Router (only for MoE layers)
    if layer_idx >= NUM_DENSE_LAYERS:
        module.router.weight.data = w["feed_forward.gate.weight"].unsqueeze(-1).unsqueeze(-1)


def populate_attn_operator_shard(module: ANEAttnOperatorShard, w: dict):
    """Fill ANEAttnOperatorShard with weights."""
    module.operator_norm.weight.data = w["operator_norm.weight"].view(HIDDEN_SIZE)
    module.ffn_norm.weight.data       = w["ffn_norm.weight"].view(HIDDEN_SIZE)

    module.q_proj.weight.data  = w["self_attn.q_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.k_proj.weight.data  = w["self_attn.k_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.v_proj.weight.data  = w["self_attn.v_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.out_proj.weight.data = w["self_attn.out_proj.weight"].unsqueeze(-1).unsqueeze(-1)

    module.q_layernorm.weight.data = w["self_attn.q_layernorm.weight"].view(HEAD_DIM)
    module.k_layernorm.weight.data = w["self_attn.k_layernorm.weight"].view(HEAD_DIM)
    module.router.weight.data = w["feed_forward.gate.weight"].unsqueeze(-1).unsqueeze(-1)


def populate_dense_layer_shard(module: ANEDenseLayerShard, w: dict, layer_idx: int):
    """Fill dense layers 0,1."""
    populate_conv_operator_shard(
        # reuse ShortConv population logic via duck-typing on the shared sub-module
        type('_', (), {
            'operator_norm': module.operator_norm,
            'short_conv': module.short_conv,
            'ffn_norm': module.ffn_norm,
            'router': None,  # no router for dense layers
        })(),
        w, layer_idx
    )
    module.operator_norm.weight.data = w["operator_norm.weight"].view(HIDDEN_SIZE)
    module.ffn_norm.weight.data       = w["ffn_norm.weight"].view(HIDDEN_SIZE)

    sc = module.short_conv
    sc.in_proj.weight.data    = w["conv.in_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    sc.conv_decode.weight.data = w["conv.conv.weight"].unsqueeze(-1)
    sc.out_proj.weight.data   = w["conv.out_proj.weight"].unsqueeze(-1).unsqueeze(-1)

    ffn = module.dense_ffn
    ffn.w1.weight.data = w["feed_forward.w1.weight"].unsqueeze(-1).unsqueeze(-1)
    ffn.w3.weight.data = w["feed_forward.w3.weight"].unsqueeze(-1).unsqueeze(-1)
    ffn.w2.weight.data = w["feed_forward.w2.weight"].unsqueeze(-1).unsqueeze(-1)


def populate_attn_decode_fixed(module: ANEAttnDecodeFixed, w: dict):
    """Fill ANEAttnDecodeFixed with weights from the layer weight dict.

    Same key names as populate_attn_operator_shard — the two classes share
    the same checkpoint key layout.
    """
    module.operator_norm.weight.data = w["operator_norm.weight"].view(HIDDEN_SIZE)
    module.ffn_norm.weight.data       = w["ffn_norm.weight"].view(HIDDEN_SIZE)

    module.q_proj.weight.data   = w["self_attn.q_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.k_proj.weight.data   = w["self_attn.k_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.v_proj.weight.data   = w["self_attn.v_proj.weight"].unsqueeze(-1).unsqueeze(-1)
    module.out_proj.weight.data = w["self_attn.out_proj.weight"].unsqueeze(-1).unsqueeze(-1)

    module.q_layernorm.weight.data = w["self_attn.q_layernorm.weight"].view(HEAD_DIM)
    module.k_layernorm.weight.data = w["self_attn.k_layernorm.weight"].view(HEAD_DIM)
    module.router.weight.data      = w["feed_forward.gate.weight"].unsqueeze(-1).unsqueeze(-1)


def populate_moe_half(module: ANEMoEHalf, w: dict, expert_start: int):
    """Fill ANEMoEHalf with per-expert weights for experts [expert_start, expert_start+N).

    Checkpoint stores weights per-expert as separate tensors:
      feed_forward.experts.{E}.w1.weight  [D, H]  gate projection (SiLU input)
      feed_forward.experts.{E}.w3.weight  [D, H]  up projection   (elementwise mul)
      feed_forward.experts.{E}.w2.weight  [H, D]  down projection
    Naming follows SwiGLU convention: output = down(silu(w1(x)) * w3(x)).
    """
    for i in range(module.N):
        e = expert_start + i
        gate_w = w[f"feed_forward.experts.{e}.w1.weight"]  # [D, H]
        up_w   = w[f"feed_forward.experts.{e}.w3.weight"]  # [D, H]
        down_w = w[f"feed_forward.experts.{e}.w2.weight"]  # [H, D]

        getattr(module, f"gate_proj_{i}").weight.data = gate_w.unsqueeze(-1).unsqueeze(-1)
        getattr(module, f"up_proj_{i}").weight.data   = up_w.unsqueeze(-1).unsqueeze(-1)
        getattr(module, f"down_proj_{i}").weight.data = down_w.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# CoreML conversion helper
# ---------------------------------------------------------------------------

def _quantize_int8(mlmodel):
    """Apply INT8 symmetric per-tensor linear quantization to all weights."""
    try:
        import coremltools as ct
        from coremltools.optimize.coreml import (
            OptimizationConfig,
            OpLinearQuantizerConfig,
            linear_quantize_weights,
        )
    except ImportError:
        raise SystemExit("coremltools not found — run with Xcode python3")

    op_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        weight_threshold=1024,
    )
    config = OptimizationConfig(global_config=op_cfg)
    return linear_quantize_weights(mlmodel, config)


def convert_module(
    module: nn.Module,
    example_inputs: tuple,
    input_specs: list,
    output_names: list,
    out_path: Path,
    do_quantize: bool = True,
) -> Path:
    """Trace, convert to CoreML, optionally quantize, save mlpackage."""
    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found — run with Xcode python3")

    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    if do_quantize:
        mlmodel = _quantize_int8(mlmodel)

    pkg_path = out_path.with_suffix(".mlpackage")
    mlmodel.save(str(pkg_path))
    print(f"  saved: {pkg_path.name}  ({pkg_path.stat().st_size // 1_048_576} MB)")
    return pkg_path


def compile_mlpackage(pkg_path: Path, out_dir: Path) -> Path:
    """Compile .mlpackage → .mlmodelc using xcrun coremlcompiler."""
    out_dir.mkdir(parents=True, exist_ok=True)
    modelc = out_dir / pkg_path.with_suffix(".mlmodelc").name
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path.resolve()), str(out_dir.resolve())],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  COMPILE ERROR: {result.stderr[:500]}", file=sys.stderr)
        return pkg_path   # return package even if compile fails
    print(f"  compiled: {modelc.name}")
    return modelc


# ---------------------------------------------------------------------------
# Per-layer conversion functions
# ---------------------------------------------------------------------------

def convert_dense_layer(layer_idx: int, w: dict, out_dir: Path, compile: bool = True):
    """Convert dense layer (0 or 1): conv operator + dense MLP."""
    print(f"[Layer {layer_idx}] dense (conv + dense MLP)")
    module = ANEDenseLayerShard(HIDDEN_SIZE, INTERMEDIATE_SIZE, CONV_L_CACHE)
    populate_dense_layer_shard(module, w, layer_idx)

    example_hidden     = torch.randn(1, HIDDEN_SIZE, 1, 1)
    example_conv_state = torch.zeros(1, HIDDEN_SIZE, CONV_L_CACHE, 1)

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found")

    pkg = convert_module(
        module,
        (example_hidden, example_conv_state),
        [
            ct.TensorType(name="hidden",     shape=example_hidden.shape),
            ct.TensorType(name="conv_state", shape=example_conv_state.shape),
        ],
        ["updated_hidden", "new_conv_state"],
        out_dir / f"lfm25_dense_layer{layer_idx}",
    )
    if compile:
        compile_mlpackage(pkg, out_dir)


def convert_conv_operator_shard(layer_idx: int, w: dict, out_dir: Path, compile: bool = True):
    """Convert conv operator shard for one MoE conv layer."""
    print(f"[Layer {layer_idx}] conv operator shard")
    module = ANEConvOperatorShard(HIDDEN_SIZE, CONV_L_CACHE, NUM_EXPERTS)
    populate_conv_operator_shard(module, w, layer_idx)

    example_hidden     = torch.randn(1, HIDDEN_SIZE, 1, 1)
    example_conv_state = torch.zeros(1, HIDDEN_SIZE, CONV_L_CACHE, 1)

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found")

    pkg = convert_module(
        module,
        (example_hidden, example_conv_state),
        [
            ct.TensorType(name="hidden",     shape=example_hidden.shape),
            ct.TensorType(name="conv_state", shape=example_conv_state.shape),
        ],
        ["updated_hidden", "new_conv_state", "ffn_normed", "routing_weights"],
        out_dir / f"lfm25_op_layer{layer_idx}",
    )
    if compile:
        compile_mlpackage(pkg, out_dir)


def convert_attn_operator_shard(
    layer_idx: int,
    w: dict,
    out_dir: Path,
    compile: bool = True,
):
    """Convert one full_attention operator shard (GQA decode, fixed MAX_SEQ cache).

    Shard budget: ~20 MB INT8 (projections: q=4MB, k=1MB, v=1MB, o=4MB; norms/router small)
    — well under 250 MB ANE ceiling.
    Shard inputs:  hidden, k_cache[MAX_SEQ], v_cache[MAX_SEQ], write_mask, attn_mask, cos, sin
    Shard outputs: updated_hidden, new_k, new_v, ffn_normed, routing_weights
    """
    print(f"[Layer {layer_idx}] attention operator shard (GQA decode, MAX_SEQ={MAX_SEQ})")
    KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 512

    module = ANEAttnDecodeFixed(
        HIDDEN_SIZE, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, NUM_EXPERTS, MAX_SEQ
    )
    populate_attn_decode_fixed(module, w)

    # Example inputs for tracing — first decode step: empty cache, write to pos 0
    example_hidden     = torch.randn(1, HIDDEN_SIZE, 1, 1)
    example_k_cache    = torch.zeros(1, KV_DIM, MAX_SEQ, 1)
    example_v_cache    = torch.zeros(1, KV_DIM, MAX_SEQ, 1)
    example_write_mask = torch.zeros(1, 1, MAX_SEQ, 1)
    example_write_mask[0, 0, 0, 0] = 1.0              # write to position 0
    example_attn_mask  = torch.full((1, 1, 1, MAX_SEQ), -1e4)
    example_attn_mask[0, 0, 0, 0]  = 0.0              # attend to position 0
    example_cos = torch.ones( 1, HEAD_DIM, 1, 1)
    example_sin = torch.zeros(1, HEAD_DIM, 1, 1)

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found")

    pkg = convert_module(
        module,
        (
            example_hidden, example_k_cache, example_v_cache,
            example_write_mask, example_attn_mask, example_cos, example_sin,
        ),
        [
            ct.TensorType(name="hidden",      shape=example_hidden.shape),
            ct.TensorType(name="k_cache",     shape=example_k_cache.shape),
            ct.TensorType(name="v_cache",     shape=example_v_cache.shape),
            ct.TensorType(name="write_mask",  shape=example_write_mask.shape),
            ct.TensorType(name="attn_mask",   shape=example_attn_mask.shape),
            ct.TensorType(name="cos",         shape=example_cos.shape),
            ct.TensorType(name="sin",         shape=example_sin.shape),
        ],
        ["updated_hidden", "new_k", "new_v", "ffn_normed", "routing_weights"],
        out_dir / f"lfm25_attn_layer{layer_idx}",
    )
    if compile:
        compile_mlpackage(pkg, out_dir)


def convert_moe_half_shard(
    layer_idx: int,
    w: dict,
    half: int,           # 0 = experts 0-15, 1 = experts 16-31
    out_dir: Path,
    compile: bool = True,
):
    """Convert one MoE expert-half shard (~177MB INT8)."""
    expert_start = half * MOE_EXPERTS_PER_HALF
    print(f"[Layer {layer_idx}] MoE half-{half} (experts {expert_start}–{expert_start+MOE_EXPERTS_PER_HALF-1})")

    module = ANEMoEHalf(HIDDEN_SIZE, MOE_INTERMEDIATE, MOE_EXPERTS_PER_HALF)
    populate_moe_half(module, w, expert_start)

    example_ffn_normed      = torch.randn(1, HIDDEN_SIZE, 1, 1)
    example_routing_weights = torch.rand(1, MOE_EXPERTS_PER_HALF, 1, 1)

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found")

    pkg = convert_module(
        module,
        (example_ffn_normed, example_routing_weights),
        [
            ct.TensorType(name="ffn_normed",      shape=example_ffn_normed.shape),
            ct.TensorType(name="routing_weights", shape=example_routing_weights.shape),
        ],
        [f"moe_contribution_half{half}"],
        out_dir / f"lfm25_moe{half}_layer{layer_idx}",
    )
    if compile:
        compile_mlpackage(pkg, out_dir)


def convert_lm_head(global_w: dict, half: int, out_dir: Path, compile: bool = True):
    """Convert one LM head vocab-half shard."""
    vocab_half = VOCAB_SIZE // 2  # 64000
    module = ANELMHeadHalf(HIDDEN_SIZE, vocab_half)

    # Slice the tied embedding weight
    lm_w = global_w["lm_head"]         # [128000, 2048]
    vocab_start = half * vocab_half
    module.proj.weight.data = lm_w[vocab_start : vocab_start + vocab_half].unsqueeze(-1).unsqueeze(-1)

    example_hidden = torch.randn(1, HIDDEN_SIZE, 1, 1)

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools not found")

    pkg = convert_module(
        module,
        (example_hidden,),
        [ct.TensorType(name="hidden", shape=example_hidden.shape)],
        [f"logits_half{half}"],
        out_dir / f"lfm25_lm_head{half}",
    )
    if compile:
        compile_mlpackage(pkg, out_dir)


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-8B-A1B to ANE CoreML shards"
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to model.safetensors (16.9 GB BF16 checkpoint)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("models/lfm25/ane"),
        help="Output directory for .mlpackage and .mlmodelc shards"
    )
    parser.add_argument(
        "--gate-only", action="store_true",
        help="Convert only layer 0 (dense) for ANE residency gate test"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Convert only this specific layer index"
    )
    parser.add_argument(
        "--skip-lm-head", action="store_true",
        help="Skip LM head conversion (for layer-by-layer testing)"
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Skip xcrun coremlcompiler step"
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}\n"
                         f"Download: huggingface-cli download LiquidAI/LFM2.5-8B-A1B model.safetensors")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compile = not args.no_compile

    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("pip install safetensors")

    print(f"Opening {args.weights} (lazy mmap)…")
    with safe_open(str(args.weights), framework="pt", device="cpu") as st:
        layers_to_convert = [args.layer] if args.layer is not None else list(range(NUM_HIDDEN_LAYERS))
        if args.gate_only:
            layers_to_convert = [0]

        for layer_idx in layers_to_convert:
            print(f"\n=== Layer {layer_idx}/{NUM_HIDDEN_LAYERS-1}: "
                  f"{LAYER_TYPES[layer_idx]} + {'dense' if layer_idx < NUM_DENSE_LAYERS else 'MoE'} ===")
            w = load_weights_for_layer(st, layer_idx, verbose=True)

            if layer_idx < NUM_DENSE_LAYERS:
                # Dense layers 0, 1: conv operator + dense MLP (single shard)
                convert_dense_layer(layer_idx, w, args.out_dir, compile=compile)
            elif LAYER_TYPES[layer_idx] == "conv":
                # Conv operator shard + 2 MoE half-shards
                convert_conv_operator_shard(layer_idx, w, args.out_dir, compile=compile)
                convert_moe_half_shard(layer_idx, w, half=0, out_dir=args.out_dir, compile=compile)
                convert_moe_half_shard(layer_idx, w, half=1, out_dir=args.out_dir, compile=compile)
            else:
                # Attention operator shard + 2 MoE half-shards
                convert_attn_operator_shard(layer_idx, w, args.out_dir, compile=compile)
                convert_moe_half_shard(layer_idx, w, half=0, out_dir=args.out_dir, compile=compile)
                convert_moe_half_shard(layer_idx, w, half=1, out_dir=args.out_dir, compile=compile)

        if not args.gate_only and not args.skip_lm_head:
            print("\n=== LM Head (tied to embed_tokens) ===")
            global_w = load_global_weights(st)
            convert_lm_head(global_w, half=0, out_dir=args.out_dir, compile=compile)
            convert_lm_head(global_w, half=1, out_dir=args.out_dir, compile=compile)

    # Write runtime metadata
    if not args.gate_only:
        meta = {
            "model": "LiquidAI/LFM2.5-8B-A1B",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_HIDDEN_LAYERS,
            "layer_types": LAYER_TYPES,
            "num_dense_layers": NUM_DENSE_LAYERS,
            "conv_L_cache": CONV_L_CACHE,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "moe_experts_per_half": MOE_EXPERTS_PER_HALF,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "vocab_size": VOCAB_SIZE,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rope_theta": ROPE_THETA,
            "tie_word_embeddings": True,
            "quant": "int8_per_tensor_linear_symmetric",
            "shard_budget_mb": 250,
            "compute_units": "ALL",
            "ane_soft_routing": True,
            "note": "Soft routing approximation: all 32 experts run with sigmoid weights"
        }
        meta_path = args.out_dir / "lfm25_runtime_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\nRuntime metadata: {meta_path}")

    print("\nDone. Run ANE residency gate:")
    print("  python3 validators/lfm25_residency_check.py --shard models/lfm25/ane/lfm25_dense_layer0.mlmodelc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
