---
layout: default
title: "Journal 053 - Phi-4-mini Twenty-Four-Layer Front Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="052-phi-4-mini-sixteen-layer-front-shard-intent.html">Previous: Journal 052</a> | <a href="054-phi-4-mini-twenty-layer-front-shard-intent.html">Next: Journal 054</a></nav>

# 2026-04-28 - Phi-4-mini Twenty-Four-Layer Front Shard Intent

**Intent**: Before the next Phi-4-mini run, test whether a very large front fused shard [0,24) can reduce model-call overhead enough to improve on the current best topology, applying the validation-first notes Iverson/APL whole-array fusion and Dragon Book call-hoisting while keeping validation ahead of performance claims.

**Setup**: Current timing context: 16+8+6+2 repeated at 17.174 tok/s, slightly ahead of 12+12+6+2 at 17.159 tok/s. Planned non-destructive probe directory: local artifacts; only consider topology 24+6+2 if build, compile, strict MLComputePlan residency, and golden validation all pass.

**Result**: Intent recorded before execution; no [0,24) artifact, compiled size, placement, golden quality, latency, energy, perplexity, or topology result yet.

**Surprise / hurdle**: This is high-risk because the artifact may be too large and larger fused ranges have possible numerical instability. The [24,32) range remains forbidden as a single 8-layer shard because prior golden validation produced NaNs despite residency.

**Lesson**: Push fusion only where the exact larger shard re-proves compile success, strict ANE residency, and golden quality; topology wins measured at 17 tok/s are too close to justify bypassing gates.

**Next**: Build/compile [0,24), run strict residency, then run golden; test 24+6+2 only if all gates pass. Do not use [24,32) as one shard, do not accept NaN/non-ANE results, and do not clean up/delete artifacts for this intent note.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="052-phi-4-mini-sixteen-layer-front-shard-intent.html">Previous: Journal 052</a> | <a href="054-phi-4-mini-twenty-layer-front-shard-intent.html">Next: Journal 054</a></nav>
