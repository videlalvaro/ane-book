# Phi-4-mini-instruct — ANE Inference

**Type**: Dense LLM  
**Params**: 3.8B  
**Speed**: ~17 tok/s decode (M4 Max, 48 GB)  
**ANE residency**: 100% (MLComputePlan verified)  
**Quantization**: INT8 per-tensor

---

## Architecture

The table below describes the 32-head / `d_head=96` export path used by
`converters/phi4_mini_rangedim_export_shard.py` and the model notes in this
README.

| Parameter | Value |
|-----------|-------|
| `d_model` | 3072 |
| `n_layers` | 32 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 (GQA) |
| `d_head` | 96 |
| `d_ff` | 8192 |
| `vocab_size` | 100352 |
| `max_seq_len` | 4096 |

Variant note: the checked-in `phi4mini_runtime_meta.json` currently describes a
different generated artifact set (`n_heads=24`, `d_head=128`, `rope_dim=96`,
`vocab_size=200064`, four LM-head shards). Do not mix dimensions from that
manifest with the 32-head export path above when doing shard-size math or runtime
validation.

## Shard Structure

The shape below is the 32-head export family described in this README. The
checked-in `phi4mini_runtime_meta.json` has a different shard list and should be
treated as its own generated artifact manifest.

```
phi4_mini_ane/
├── layer_00.mlmodelc/ ... layer_31.mlmodelc/    # 32 attention+FFN shards
├── lm_head_0.mlmodelc/ lm_head_1.mlmodelc/      # 2 vocab-half LM head shards
├── phi4_mini_embed.bin                           # float16 embedding, 100352 × 3072
└── phi4mini_runtime_meta.json                    # shape manifest
```

Earlier 3-layer INT8 Phi-4-mini shards validated near the practical ANE ceiling.
Use the compiled `.mlmodelc` size for the artifact family you are reproducing;
do not mix shard-count or LM-head-split numbers between manifests.

## Building

```bash
# Requires: GGUF file at models/phi4-mini/Phi-4-mini-instruct.Q8_0.gguf
# Uses Xcode python3 (coremltools 9)

# Export all layer shards (RangeDim T=1..4)
/usr/bin/python3 ../../converters/phi4_mini_rangedim_export_shard.py --all

# Export LM head shards
/usr/bin/python3 ../../converters/phi4_mini_lm_head_shards.py

# Validate residency (all shards must be 100% ANE)
/usr/bin/python3 ../../validators/phi4_mini_residency_check.py
```

## Runtime

Swift host: `runtime/phi4_mini_ane.swift`

Features:
- Stateful KV cache (MLState, zero host copy)
- RangeDim T=1..4 (n-gram speculative decode)
- Chat mode with system prompt support
- Streaming token output

## Golden Validation

```bash
# Capture PyTorch FP16 golden (in .venv313 with HuggingFace)
python3 -c "..."   # see validators/phi4_mini_residency_check.py

# Compare CoreML vs golden
python3 ../../validators/compare_logits.py \
    --golden phi4mini_golden.npy \
    --coreml phi4mini_coreml_out.npy
# Expected: cos ≥ 0.999
```

## Download

The GGUF source: `microsoft/Phi-4-mini-instruct` on Hugging Face.  
Quantized GGUF: `bartowski/Phi-4-mini-instruct-GGUF` (use Q8_0).
