# Hy-MT 1.5 — ANE Inference (Translation)

**Type**: Dense LLM, specialized for translation  
**Params**: ~1.8B  
**Speed**: ~34 tok/s decode (M4 Max, 48 GB)  
**ANE residency**: 100% (MLComputePlan verified)  
**Quantization**: INT8 per-tensor

---

## Overview

Hy-MT 1.5 is a fine-tuned translation model. The ANE port achieves the highest
throughput of any model in this collection (~34 tok/s) due to its compact size
and efficient GQA attention.

## Shard Structure

```
hymt_ane/
├── layer_00.mlmodelc/ ... layer_N.mlmodelc/   # per-layer shards
├── lm_head_0.mlmodelc/                         # LM head (single shard)
├── hymt_embed.bin                              # float16 embedding
└── hymt_runtime_meta.json                      # shape manifest
```

## Building

```bash
# Export all shards
/usr/bin/python3 ../../converters/hymt_export_runtime.py

# Export with RangeDim (variable T)
/usr/bin/python3 ../../converters/hymt_rangedim_export_shard.py --all

# Validate parity with HuggingFace reference
/usr/bin/python3 ../../validators/hymt_rangedim_parity_check.py
```

## Runtime

Swift host: `runtime/hymt_ane.swift`

Features:
- Stateful KV cache (MLState)
- RangeDim T=1..4
- Translation mode: source language → target language prompt template

## Download

Source model: `HuggingFaceTB/Hy-MT-1.5` on Hugging Face.  
Export from GGUF: use `converters/hymt_export_runtime.py`.
