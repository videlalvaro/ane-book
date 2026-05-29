---
layout: default
title: "Journal 021 - Phi-4-mini LM Head Shards Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="020-phi-4-mini-lm-head-shard-builder-intent.html">Previous: Journal 020</a> | <a href="022-phi-4-mini-runtime-scaffolding-smoke-passed.html">Next: Journal 022</a></nav>

# 2026-04-27 - Phi-4-mini LM Head Shards Passed

**Intent**: Move the compute-heavy Phi-4-mini final RMSNorm plus tied LM head projection onto ANE as sharded CoreML artifacts, following the ANE-only mandate and validation-before-scale discipline.

**Setup**: Implemented [converters/phi4_mini_lm_head_shards.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_lm_head_shards.py) and the LM-head golden validator; added `lm-head-shard` and `lm-head` stages to the Phi orchestration script. Built 4 INT8 CoreML shards under local artifacts; shard 0 was built and gated first, then shards 1–3 were built. Final command: the LM-head stage of the Phi orchestration script.

**Result**: PASS. Verification found 4/4 `.mlpackage` and 4/4 `.mlmodelc`; total LM-head artifact directory size was 1.1G. All 4 residency reports passed with conv_total=1 conv_ane=1 conv_non_ane=0 and compute_total=8 compute_ane=8 compute_non_ane=0. Numerical smoke: shard0 cos=0.9998542691 rmse=0.1045568064 max_abs=0.7450714111; shard1 cos=0.9998624309 rmse=0.0728808641 max_abs=0.7602825165; shard2 cos=0.9998748088 rmse=0.0604509786 max_abs=0.7003631592; shard3 cos=0.9998865075 rmse=0.0523771085 max_abs=0.5423495770. Aggregate cos min=0.9998542691 max=0.9998865075 mean=0.9998695041.

**Surprise / hurdle**: The LM head had to be validated shard 0 first before scaling to the remaining vocab shards, preserving the same no-fallback policy used for layer shards.

**Lesson**: The Phi-4-mini final projection can be split into four INT8 ANE-resident CoreML shards with high numerical agreement instead of remaining a host-side compute path.

**Next**: No performance or energy benchmarking was run, and no cleanup or deletion was performed; next gated work is full-runtime integration or end-to-end logits validation.

**Refs**: [converters/phi4_mini_lm_head_shards.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_lm_head_shards.py); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="020-phi-4-mini-lm-head-shard-builder-intent.html">Previous: Journal 020</a> | <a href="022-phi-4-mini-runtime-scaffolding-smoke-passed.html">Next: Journal 022</a></nav>
