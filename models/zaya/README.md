# ZAYA1-8B — ANE Inference (MoE LLM)

**Type**: Mixture of Experts LLM  
**Params**: 8B (1.58-bit average from binarized training)  
**Speed**: ~9 tok/s decode (M4 Max, 48 GB)  
**ANE residency**: 100% (MLComputePlan verified)  
**Quantization**: INT8 per-tensor (post-conversion)  
**Routing**: Soft (all 8 experts run, weighted sum)

---

## Artifact Note

This README describes the 8-expert, 28-layer ZAYA runtime/book variant. The
checked-in `converters/zaya_full_convert.py` script is an Exp 34 RangeDim exporter
for a different ZAYA artifact family: hidden size 2048, 16 experts plus one null
router slot, and MoE layers `1,3,...,79` (40 MoE shards). Keep the README/runtime
manifest and converter family together when reproducing results.

---

## Architecture

| Parameter | Value |
|-----------|-------|
| `d_model` | 4096 |
| `n_layers` | 28 MoE layers |
| `n_heads` | 32 |
| `n_kv_heads` | 8 (GQA) |
| `d_head` | 128 |
| `n_experts` | 8 |
| `top_k_routing` | 2 (soft-routed to all 8 in CoreML) |
| `vocab_size` | 32000 |

## Why Soft Routing?

CoreML graphs cannot branch on runtime values (no `if` on tensor content).
All 8 experts run every step with the router's softmax weights. Non-selected
experts contribute near-zero outputs. Quality is preserved because the top-2
weights dominate.

This trades compute (8× expert work vs 2×) for a fully-static graph that stays
100% ANE-resident. At 9 tok/s on M4 Max, the overhead is acceptable.

## Shard Structure (58 compiled model shards)

```
zaya_ane/
├── attn/
│   └── zaya_attn_{00..27}.mlmodelc   # 28 attention shards
├── moe_rangedim/
│   └── zaya_moe_{00..27}.mlmodelc    # 28 MoE FFN shards (8 experts packed)
├── lm_head/
│   ├── zaya_lm_head_0.mlmodelc       # vocab[:16000]
│   └── zaya_lm_head_1.mlmodelc       # vocab[16000:]
├── zaya_embed.bin                     # float16 embedding, 32000 × 4096
└── zaya_runtime_meta.json
```

Each MoE shard contains all 8 expert FFNs. The router output (softmax weights)
is computed in the same shard and the weighted sum is applied before returning.

## Building

```bash
# Download ZAYA1 weights (requires HuggingFace account)
# Model: "1bitLLM/bitnet_b1_58-large" or your ZAYA1-8B checkpoint

# Export the 8-expert variant with the converter/manifest that matches this
# README. Do not use converters/zaya_full_convert.py for these dimensions; that
# checked-in script documents the Exp 34 16-expert RangeDim path.

# Validate residency on MoE shards
/usr/bin/python3 ../../validators/phi4_mini_residency_check.py \
    --shard-dir zaya_ane/moe_rangedim/

# Validate quality
/usr/bin/python3 ../../validators/zaya_golden_validator.py
# Expected: cos ≥ 0.97
```

## Runtime

Swift host: `runtime/zaya_ane.swift`

Features:
- Stateful KV cache (MLState, per attention shard)
- RangeDim T=1..4 on MoE shards
- Chat mode (LLaMA-2 prompt format)
- Two-pass LM head (vocab-half shards, concatenate logits)

## Key Decision: Attention-MoE Split

Attention shards and MoE shards are separate `.mlmodelc` files rather than
combining all of a transformer layer into one package. This is because:

1. A full MoE layer (attention + 8 experts) would exceed the 250 MB shard limit
2. Separate shards allow independent residency validation
3. The Swift runtime alternates: `attn[i](state) → moe[i] → attn[i+1](state) → ...`

## Download

Source: `1bitLLM/bitnet_b1_58-large` (or equivalent ZAYA1-8B checkpoint).
Use the converter that matches the manifest family you are reproducing; the
checked-in `converters/zaya_full_convert.py` documents the Exp 34 16-expert path.
