---
layout: default
title: "Journal 008 - Phi-4-mini Layers 2–3 Bounded Smoke Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="007-phi-4-mini-layers-2-3-bounded-smoke-intent.html">Previous: Journal 007</a> | <a href="009-phi-4-mini-bounded-run-range-orchestration-intent.html">Next: Journal 009</a></nav>

# 2026-04-27 - Phi-4-mini Layers 2–3 Bounded Smoke Passed

**Intent**: Record the bounded layers 2–3 Phi-4-mini-instruct guarded outcome before any scale-out, following the established quality-before-performance workflow and validation discipline.

**Setup**: Ran only layers 2 and 3 through INT8 full-layer mlpackage conversion, compile, strict MLComputePlan residency, and per-layer PyTorch-vs-CoreML numerical smoke. temporary JSON outputs.

**Result**: PASS. Layers 2 and 3 INT8 full-layer mlpackage conversion succeeded; compile succeeded; each mlpackage/mlmodelc was 96M. Strict residency passed for both: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Numerical smoke passed: layer 2 cos(hidden)=0.999893, rmse=0.008980, max_abs=0.058594; layer 3 cos(hidden)=0.999878, rmse=0.010047, max_abs=0.090942.

**Surprise / hurdle**: The two-layer batch stayed bounded and preserved strict ANE residency with no observed non-ANE compute ops.

**Lesson**: Phi-4-mini INT8 full-layer conversion remains repeatably ANE-resident and numerically close through layers 2 and 3.

**Next**: No full conversion, perf, energy, or cleanup was run; proceed only through the gated scale-out flow.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="007-phi-4-mini-layers-2-3-bounded-smoke-intent.html">Previous: Journal 007</a> | <a href="009-phi-4-mini-bounded-run-range-orchestration-intent.html">Next: Journal 009</a></nav>
