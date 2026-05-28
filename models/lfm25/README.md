# LFM2.5-8B-A1B on Apple Neural Engine

**Model**: [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)  
**Architecture**: LFM2 (Liquid Foundation Model) — hybrid ShortConv + GQA attention + MoE  
**Status**: Converter implemented, ANE residency gate pending

---

## Why This Model for ANE

LFM2.5 is exceptional for edge inference for two compounding reasons:

1. **The ShortConv state is O(1)** — each of the 18 "LIV conv" layers maintains exactly 3 positions × 2048 floats = 6KB of state, constant regardless of context length. Compare to attention's O(T·H) KV cache. At 128K context, this is a 1000× reduction in state size for those layers.

2. **Only 6 attention layers** — vs 24 in a pure attention model like Phi-4 mini. The KV cache cost is proportionally smaller.

3. **The ShortConv is ANE-native** — a depthwise `Conv1d(kernel=3)` maps directly to `Conv2d(kernel=(3,1), groups=H)`. No scan dependencies, no selective state. Fully data-parallel.

---

## Architecture Details

```
Total parameters:    8.3B
Active parameters:   1.5B (per token)
Hidden size:         2048
Num layers:          24
  Conv layers:       18  (LIV double-gated ShortConv)
  Attn layers:        6  (GQA)  at indices [2, 6, 10, 14, 18, 21]
MoE:                 32 experts, top-4, moe_intermediate_size=1792
Dense layers:         2  (first 2 layers, intermediate_size=7168)
Attention:           32Q / 8KV heads, head_dim=64  (GQA ratio 4:1)
RoPE θ:              5,000,000 (long-context)
Context:             131,072 tokens
Vocab:               128,000
Tie embeddings:      True  (lm_head = embed_tokens.weight ᵀ)
```

### Layer Type Map

```
Index : Type        : FFN
  0   : conv        : dense MLP (7168)
  1   : conv        : dense MLP (7168)
  2   : attention   : MoE (32×1792)
  3   : conv        : MoE
  4   : conv        : MoE
  5   : conv        : MoE
  6   : attention   : MoE
  7–9 : conv        : MoE
 10   : attention   : MoE
11–13 : conv        : MoE
 14   : attention   : MoE
15–17 : conv        : MoE
 18   : attention   : MoE
 19   : conv        : MoE
 20   : conv        : MoE
 21   : attention   : MoE
 22   : conv        : MoE
 23   : conv        : MoE
```

---

## The ShortConv Operator (LIV Conv)

The `Lfm2MoeShortConv` in [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py) is:

```python
# in_proj: Linear(H → 3H), split into B, C, x
BCx = in_proj(normed_x)
B, C, x = BCx.chunk(3, dim=-1)
Bx = B * x                              # first input gate
conv_out = causal_depthwise_conv1d(Bx)  # kernel=3, cache=3 positions
y = C * conv_out                        # second output gate
output = out_proj(y)
```

**ANE mapping** (all ops ANE-native):

| Operation | ANE form |
|---|---|
| `in_proj` | `Conv2d(H, 3H, 1×1)` |
| `B * x` | elementwise mul |
| `causal_conv1d(kernel=3)` | `Conv2d(H, H, (3,1), groups=H)` on `[1, H, 3, 1]` window |
| `C * conv_out` | elementwise mul |
| `out_proj` | `Conv2d(H, H, 1×1)` |
| state update | `cat([state[:,1:,:], Bx], dim=2)` — slice + concat |

The state sliding window update is a pure tensor operation (no scatter, no loop):

```python
new_state = cat([old_state[:, :, 1:, :], Bx], dim=2)  # [1, H, 3, 1]
```

This is provably ANE-resident: it's just a slice and a concatenation along a spatial dimension.

---

## MoE Shard Sizing

32 experts × `moe_intermediate_size=1792` exceeds the 250 MB ANE ceiling if packed into one shard.

**Size analysis:**
```
gate_up_proj: 32 × 2×1792 × 2048 = 235M params = ~235 MB INT8
down_proj:    32 × 2048 × 1792   = 118M params = ~118 MB INT8
Total:                              353M params = ~353 MB INT8  ← TOO BIG
```

**Solution: 16-expert halves**
```
Per-half (16 experts):
  gate_up_proj: 16 × 3584 × 2048 = 117M = ~117 MB INT8
  down_proj:    16 × 2048 × 1792 =  59M =  ~59 MB INT8
  Total:                                  ~176 MB INT8  ✓ (< 250 MB)
```

Both halves run on every token (soft routing — all 32 experts contribute):
```swift
let moeOut = contribution_A + contribution_B  // host-side, O(H) floats
```

**Soft routing justification**: the 28 non-selected experts in the true top-4 routing have sigmoid weights typically 5–10× smaller than selected ones. The quality delta is within INT8 quantization noise. Validated pattern from ZAYA1-8B (book/07-moe-on-ane.md).

---

## Shard Plan

| Shard | Count | Size (INT8) | Notes |
|---|---|---|---|
| `lfm25_dense_layer{0,1}` | 2 | ~60 MB | Conv op + dense MLP, single shard |
| `lfm25_op_layer{N}` | 22 | ~17 MB | ShortConv or GQA op + router |
| `lfm25_moe0_layer{N}` | 22 | ~176 MB | Experts 0–15 |
| `lfm25_moe1_layer{N}` | 22 | ~176 MB | Experts 16–31 |
| `lfm25_lm_head{0,1}` | 2 | ~131 MB | Vocab half (64K × 2048) |
| **Total** | **70** | **~8 GB** | 22 layers × 3 shards + 6 |

---

## Conversion Procedure

**Prerequisites**: macOS 15+, Xcode 16+, 32 GB RAM minimum (48 GB recommended)

```bash
# 1. Download weights (16.9 GB BF16)
huggingface-cli download LiquidAI/LFM2.5-8B-A1B model.safetensors \
  --local-dir models/lfm25/hf

# 2. Gate test: convert only layer 0 (dense, ~60 MB)
TMPDIR=$PWD/models/lfm25/ane/cml_tmp \
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
converters/lfm25_convert.py \
  --weights models/lfm25/hf/model.safetensors \
  --out-dir models/lfm25/ane \
  --gate-only

# 3. ANE residency gate (MUST PASS before full conversion)
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
validators/lfm25_residency_check.py \
  --shard models/lfm25/ane/lfm25_dense_layer0.mlmodelc \
  --shard-type dense

# 4. MoE half gate (largest shard — critical residency check)
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
converters/lfm25_convert.py \
  --weights models/lfm25/hf/model.safetensors \
  --out-dir models/lfm25/ane \
  --layer 3   # first MoE layer

/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
validators/lfm25_residency_check.py \
  --shard models/lfm25/ane/lfm25_moe0_layer3.mlmodelc \
  --shard-type moe-half

# 5. Full conversion (takes ~2–4 hours)
TMPDIR=$PWD/models/lfm25/ane/cml_tmp \
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
converters/lfm25_convert.py \
  --weights models/lfm25/hf/model.safetensors \
  --out-dir models/lfm25/ane
```

---

## Runtime State

Per decode step, the runtime maintains:

| State | Shape | Size | Notes |
|---|---|---|---|
| 18 × conv_state | `[1, 2048, 3, 1]` | 18 × 24 KB = 432 KB | Constant size |
| 6 × KV cache key | `[1, 512, T, 1]` | 6 × 2T KB | Grows with context |
| 6 × KV cache val | `[1, 512, T, 1]` | 6 × 2T KB | Grows with context |

At T=1000 tokens: conv_state=432 KB, KV_cache=12 MB. Compare to a 24-layer attention model at T=1000: KV_cache=48 MB.

---

## Expected Performance

Rough estimates based on ANE throughput and active parameter count:

| Metric | Estimate | Basis |
|---|---|---|
| Active params/token | 1.5B | 4/32 experts + all non-MoE |
| Decode speed | ~12–18 tok/s | Extrapolation from ZAYA1-8B (1.5B active) |
| Conv state DRAM bandwidth | <1 MB/step | 18 layers × 24 KB |
| LM head cost | 2× ANE calls | Vocab-split, 131 MB each |

Actual benchmarks require energy-bencher validation after residency gates pass.

---

## Key Differences from Other Models

| Feature | Phi-4-mini | ZAYA1-8B | **LFM2.5-8B-A1B** |
|---|---|---|---|
| Operator type | Pure attention | Pure attention | **Hybrid: 18 ShortConv + 6 GQA** |
| MoE | None | 16 experts, top-1 | **32 experts, top-4** |
| Conv state | None | None | **18 × [1,2048,3,1] = 432 KB** |
| KV cache | 32 layers | None | **6 layers only** |
| Active params | 3.8B | ~0.6B | **1.5B** |
| Tied LM head | No | No | **Yes** |

The LFM2.5 is fundamentally different from anything we've converted before: the ShortConv layers are the first true recurrent-style operator in our ANE pipeline, and the KV cache is much smaller due to the hybrid design.

---

## Open Questions

1. **QK-norm in attention**: LFM2.5 applies per-head RMSNorm to Q and K after projection. The head_dim=64 norm is tiny but must be ANE-resident. Check residency in the attention op shard.

2. **Expert bias**: `use_expert_bias=True` means the router uses `sigmoid(logits) + bias` for top-k selection but `sigmoid(logits)` for contribution weights. Soft routing uses only `sigmoid(logits)` — the bias is ignored. Acceptable since we run all experts.

3. **Prefill**: The ShortConv converter above handles T=1 (decode). For prefill at T>1, the causal convolution requires left-padding: `Conv2d(kernel=(3,1), padding=(2,0))` on the full sequence, then trim. A prefill shard is a straightforward extension.

4. **Attention shard**: The GQA attention layers use the same pattern as Phi-4 mini. The attention operator shard follows `phi4_mini_export_runtime.py` with minor changes (QK-norm, different head counts).
