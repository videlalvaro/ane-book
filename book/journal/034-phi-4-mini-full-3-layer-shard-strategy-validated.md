---
layout: default
title: "Journal 034 - Phi-4-mini Full 3-Layer Shard Strategy Validated"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="033-llm-int8-ane-conv2d-adaptation-review-intent.html">Previous: Journal 033</a> | <a href="035-phi-4-mini-fused-runtime-migration-intent.html">Next: Journal 035</a></nav>

# 2026-04-27 - Phi-4-mini Full 3-Layer Shard Strategy Validated

**Intent**: Record the completed full-model validation of the Phi-4-mini 3-layer fused-shard strategy, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Validated ranges [0,3), [3,6), [6,9), [9,12), [12,15), [15,18), [18,21), [21,24), [24,27), [27,30), and tail [30,32) in local artifacts; no cleanup/deletion and no energy benchmarking.

**Result**: PASS. All 11 compiled `.mlmodelc` artifacts are present; total directory size is 6.0G. Strict residency passed all ranges: 3-layer ranges conv=12/12/0 and compute=438/438/0; tail conv=8/8/0 and compute=293/293/0. Golden smoke passed all ranges: cos_hidden min=0.99952924 mean=0.99964909 max=0.99976834; max RMSE=0.09616414; max_abs=0.47656250.

**Surprise / hurdle**: Every planned fused range, including the 2-layer tail, compiled and remained ANE-resident despite the earlier shard-size caution from the first 288M 3-layer probe.

**Lesson**: The 3-layer Phi-4-mini fused strategy is validated across the full model for ANE residency and numerical smoke, so the next bottleneck is runtime migration rather than more per-range proof.

**Next**: Proceed to fused-shard manifest/runtime migration if desired; do not run energy benchmarking unless explicitly requested.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="033-llm-int8-ane-conv2d-adaptation-review-intent.html">Previous: Journal 033</a> | <a href="035-phi-4-mini-fused-runtime-migration-intent.html">Next: Journal 035</a></nav>
