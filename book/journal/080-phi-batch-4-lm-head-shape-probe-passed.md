---
layout: default
title: "Journal 080 - Phi Batch-4 LM-Head Shape Probe Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="079-phi-lm-head-top-k-shard-residency-failed.html">Previous: Journal 079</a> | <a href="081-phi-batch-4-lm-head-full-set-gated.html">Next: Journal 081</a></nav>

# 2026-04-28 - Phi Batch-4 LM-Head Shape Probe Passed

**Intent**: Test the Iverson/APL-style fatter-array direction on the LM head before attempting any full runtime migration: score multiple hidden vectors with one ANE 1x1 conv instead of issuing one CoreML prediction per token.

**Setup**: Extended [converters/phi4_mini_lm_head_shards.py](https://github.com/videlalvaro/ane-models/blob/main/converters/phi4_mini_lm_head_shards.py) with opt-in `--batch-tokens` while preserving the single-token default. Built only representative shard 0 with `--batch-tokens 4` into `lm_head_shards_bt4`, then extended the LM-head golden validator for batched validation and added the LM-head batch benchmark for a shard-local microbench.

**Result**: The batch-4 shard passed strict residency: `conv_total=1`, `conv_ane=1`, `conv_non_ane=0`, `compute_total=8`, `compute_ane=8`, `compute_non_ane=0`, `PASS=True`. Golden passed against NumPy with `cos_logits=0.999926`, `rmse=0.103638`, `max_abs=0.812080`. Microbench over 100 measured iterations showed `single_ms_per_token=1.608`, `batch_ms_per_token=0.691`, `batch_ms_per_call=2.764`, `speedup_per_token=2.327` for shard 0.

**Surprise / hurdle**: The shape is ANE-resident and materially faster per token, but it needs multiple independent hidden vectors. It is a multi-stream, speculative verification, or prefill-like throughput lever, not an automatic greedy single-stream decode win.

**Lesson**: Fattening the spatial token dimension is a valid ANE shape for the LM head; batching can amortize CoreML/ANE submission and reuse weight movement without introducing CPU compute.

**Next**: Do not replace the production runtime yet. Scale this only after deciding which workload supplies the independent hidden vectors: multi-agent batching, speculative draft verification, or batched prefill/head scoring.

**Refs**: [converters/phi4_mini_lm_head_shards.py](https://github.com/videlalvaro/ane-models/blob/main/converters/phi4_mini_lm_head_shards.py); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="079-phi-lm-head-top-k-shard-residency-failed.html">Previous: Journal 079</a> | <a href="081-phi-batch-4-lm-head-full-set-gated.html">Next: Journal 081</a></nav>
