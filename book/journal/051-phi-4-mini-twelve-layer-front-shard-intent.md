---
layout: default
title: "Journal 051 - Phi-4-mini Twelve-Layer Front Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="050-phi-4-mini-eight-layer-asymmetric-fusion-outcome.html">Previous: Journal 050</a> | <a href="052-phi-4-mini-sixteen-layer-front-shard-intent.html">Next: Journal 052</a></nav>

# 2026-04-28 - Phi-4-mini Twelve-Layer Front Shard Intent

**Intent**: Before the next Phi-4-mini optimization run, probe a larger front fused shard for layers [0,12) to reduce layer model-call count beyond the asymmetric 8+8+8+6+2 topology, applying the validation-first notes Iverson/APL whole-array fused-operator thinking while preserving validation-before-scale discipline.

**Setup**: Current timing context: asymmetric 8+8+8+6+2 reached about 16.65 tok/s with layer time about 55 ms/token. Planned non-destructive probe directory: local artifacts; build and compile only the INT8 stateful CoreML shard [0,12) before any broader topology work.

**Result**: Intent recorded before execution; no 12-layer artifact, compiled size, placement, latency, energy, cosine, RMSE, max_abs, perplexity, or manifest results yet.

**Surprise / hurdle**: Do not assume success from the 8-layer results: [24,32) was strict ANE-resident but produced NaNs in golden validation, so residency alone is not enough for a larger fused shard. The ANE_CHAIN_SCHEMA empirical shard-size law must also be revalidated for this larger shard rather than extrapolated from prior 6-layer or 8-layer ranges.

**Lesson**: Larger whole-array layer fusion is useful only when the exact larger shard re-proves compile success, strict ANE residency, and golden quality; empirical size and numerical behavior do not safely extrapolate.

**Next**: Gate order is build/compile [0,12), run strict MLComputePlan residency, then run range golden; only if those pass consider [12,24) and an asymmetric 12+12+6+2 manifest. Do not run profiling, energy benchmarking, cleanup, deletion, or code changes before the gates pass.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="050-phi-4-mini-eight-layer-asymmetric-fusion-outcome.html">Previous: Journal 050</a> | <a href="052-phi-4-mini-sixteen-layer-front-shard-intent.html">Next: Journal 052</a></nav>
