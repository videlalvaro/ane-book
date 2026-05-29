---
layout: default
title: Production LLM Inference on the Apple Neural Engine
---

# Production LLM Inference on the Apple Neural Engine

A practitioner's guide to running production LLM inference on the Apple Neural
Engine with CoreML, Swift runtimes, ANE-only residency checks, and validated
model manifests.

## Chapters

| Chapter | Topic |
|---------|-------|
| [00 - Why ANE?](00-why-ane.html) | ANE vs GPU vs CPU; why CoreML; the Conv2d trick |
| [01 - ANE Laws](01-ane-laws.html) | Empirical rules: shard limits, quantization, residency |
| [02 - Porting Recipe](02-porting-recipe.html) | GGUF to CoreML, step by step |
| [03 - Quantization](03-quantization.html) | INT8 production, INT4 tradeoffs, the silent CPU fallback |
| [04 - Shard Sizing](04-shard-sizing.html) | Layer count vs size, 250 MB limit, LM-head splits |
| [05 - Stateful KV Cache](05-stateful-kv-cache.html) | MLState, Swift daemon design, decode loop |
| [06 - RangeDim + Speculative](06-rangedim-speculative.html) | Variable T, n-gram acceptance |
| [07 - MoE on ANE](07-moe-on-ane.html) | Soft routing, per-expert dispatch, ZAYA and Privacy Filter |
| [08 - Experiment Log](08-experiments.html) | Experiments, results, and lessons learned |
| [09 - Decision Journal](09-journal.html) | The thinking behind the hard calls |

## Repository

The source code, converters, Swift runtimes, validators, and model manifests live
in the [ane-models repository](https://github.com/videlalvaro/ane-models).