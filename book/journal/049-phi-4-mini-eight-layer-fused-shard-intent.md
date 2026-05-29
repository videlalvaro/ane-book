---
layout: default
title: "Journal 049 - Phi-4-mini Eight-Layer Fused Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="048-phi-4-mini-lm-head-optimization-outcome.html">Previous: Journal 048</a> | <a href="050-phi-4-mini-eight-layer-asymmetric-fusion-outcome.html">Next: Journal 050</a></nav>

# 2026-04-28 - Phi-4-mini Eight-Layer Fused Shard Intent

**Intent**: Probe whether a larger 8-layer Phi-4-mini fused INT8 stateful CoreML shard can reduce the dominant layer-chain cost after LM-head top-k failed strict ANE residency and 3-way/8-way full-logit LM-head sharding did not improve throughput. The hypothesis follows the validation-first notes Iverson/APL whole-array and fused-operator thinking: treat a larger contiguous layer range as one fused array operation instead of optimizing the now-smaller LM-head path.

**Setup**: Current timing context: layer execution dominates decode at about 57-60 ms/token, while the LM head remains about 5.1 ms/token. Planned non-destructive probe directory: local artifacts. Build only the first range [0,8) as an INT8 stateful CoreML shard, compile it, then run strict MLComputePlan residency and range golden quality before any scale-out or profiling.

**Result**: Intent recorded before execution; no 8-layer artifact, compiled size, residency placement, golden cosine/RMSE/max_abs, latency, energy, or perplexity numbers yet.

**Surprise / hurdle**: Prior 5-layer and 6-layer fusion exceeded older conservative shard-size guidance while remaining ANE-resident, but the empirical ANE_CHAIN_SCHEMA shard-size law must be revalidated for this larger range rather than assumed from earlier ranges.

**Lesson**: When LM-head variants stop moving throughput and layers dominate, the next fused-layer size is a valid hypothesis only after the first range re-proves compile success, strict ANE residency, and golden quality.

**Next**: Build/compile only [0,8) under local artifacts; run strict MLComputePlan residency and range golden; do not scale out, profile, benchmark energy, clean up, or delete artifacts until those gates pass.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="048-phi-4-mini-lm-head-optimization-outcome.html">Previous: Journal 048</a> | <a href="050-phi-4-mini-eight-layer-asymmetric-fusion-outcome.html">Next: Journal 050</a></nav>
