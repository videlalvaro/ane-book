---
layout: default
title: "Journal 037 - Phi-4-mini Four-Layer Fused Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="036-phi-4-mini-fused-runtime-migration-outcome.html">Previous: Journal 036</a> | <a href="038-phi-4-mini-four-layer-fused-shard-residency-passed.html">Next: Journal 038</a></nav>

# 2026-04-27 - Phi-4-mini Four-Layer Fused Shard Intent

**Intent**: Probe whether a larger fused Phi-4-mini shard beyond the validated 3-layer/288M pattern remains ANE-resident before any broader scale-out, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Planned non-destructive build: one INT8 stateful CoreML shard for layers [0,4) under local artifacts, then compile and run strict MLComputePlan residency validation. No cleanup/deletion, no golden validation unless requested after residency, and no energy benchmarking.

**Result**: Intent recorded before execution; no placement, latency, energy, cosine, perplexity, or artifact-size results yet.

**Surprise / hurdle**: The open question is whether fusing a fourth layer crosses a CoreML/ANE placement boundary even though smaller fused shards have stayed resident.

**Lesson**: Fused-shard scale-out should advance only after the next larger representative shard proves strict ANE residency.

**Next**: Build and compile only the layers [0,4) probe shard, then record strict residency results before considering golden validation or larger fused ranges.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="036-phi-4-mini-fused-runtime-migration-outcome.html">Previous: Journal 036</a> | <a href="038-phi-4-mini-four-layer-fused-shard-residency-passed.html">Next: Journal 038</a></nav>
