---
layout: default
title: "Journal 012 - Phi-4-mini Layers 4–7 Bounded Run-Range Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="011-phi-4-mini-layers-4-7-bounded-run-range-intent.html">Previous: Journal 011</a> | <a href="013-phi-4-mini-four-layer-batch-runner-intent.html">Next: Journal 013</a></nav>

# 2026-04-27 - Phi-4-mini Layers 4–7 Bounded Run-Range Passed

**Intent**: Record the actual bounded Phi-4-mini layers 4–7 build+test outcome, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Ran the bounded layer-range stage of the Phi orchestration script; per-layer flow: preflight, convert, compile, strict MLComputePlan residency, and numerical smoke.

**Result**: PASS. Layers 4, 5, 6, and 7 converted and compiled successfully. For each layer, mlpackage=96M, mlmodelc=96M, and meta=4.0K. Strict MLComputePlan residency passed for each layer: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Numerical smoke passed: L4 cos=0.999892 rmse=0.012303 max_abs=0.044922; L5 cos=0.999893 rmse=0.012867 max_abs=0.101562; L6 cos=0.999875 rmse=0.012494 max_abs=0.058105; L7 cos=0.999883 rmse=0.012904 max_abs=0.042969.

**Surprise / hurdle**: The first real four-layer `run-range` stayed bounded while preserving strict ANE residency and numerical-smoke gates for every layer.

**Lesson**: The bounded run-range runner can advance four Phi-4-mini INT8 full-layer shards at a time without observed ANE fallback or numerical-smoke regression.

**Next**: No full 32-layer conversion, performance run, energy run, cleanup, or deletion was run; continue only through gated bounded ranges.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="011-phi-4-mini-layers-4-7-bounded-run-range-intent.html">Previous: Journal 011</a> | <a href="013-phi-4-mini-four-layer-batch-runner-intent.html">Next: Journal 013</a></nav>
