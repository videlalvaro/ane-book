---
layout: default
title: "Journal 046 - Phi-4-mini Six-Layer Fused Strategy Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="045-phi-4-mini-five-layer-fused-strategy-intent.html">Previous: Journal 045</a> | <a href="047-phi-4-mini-five-and-six-layer-fusion-outcome.html">Next: Journal 047</a></nav>

# 2026-04-28 - Phi-4-mini Six-Layer Fused Strategy Intent

**Intent**: After the 5-layer fused runtime best observed 15.661 tok/s, test whether a 6-layer Phi-4-mini fused topology can push decode throughput higher, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Existing non-destructive probe: single INT8 fused shard [0,6) built and compiled under local artifacts; artifact size 577M. Planned full topology: [0,6), [6,12), [12,18), [18,24), [24,30), and tail [30,32). Run strict MLComputePlan residency and range golden across all ranges before generating any 6-layer runtime manifest/profile.

**Result**: Intent recorded after representative probe passed. Probe strict residency passed: conv_total=24 conv_ane=24 conv_non_ane=0; compute_total=873 compute_ane=873 compute_non_ane=0. Probe range golden passed: cos_hidden=0.999451, rmse_hidden=0.031468, max_abs_hidden=0.500000.

**Surprise / hurdle**: The 577M compiled shard exceeds prior fused-shard sizes yet remains fully ANE-resident for [0,6), so every remaining range must independently prove compile success, strict residency, and numerical quality before runtime migration.

**Lesson**: A larger fused shard can be considered only as a validated topology, not from a single representative pass, because compiled size and placement remain empirical.

**Next**: Build/compile the remaining 6-layer ranges under local artifacts, then run strict residency and range golden for all ranges; do not delete/clean up artifacts and do not run energy benchmarking.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="045-phi-4-mini-five-layer-fused-strategy-intent.html">Previous: Journal 045</a> | <a href="047-phi-4-mini-five-and-six-layer-fusion-outcome.html">Next: Journal 047</a></nav>
