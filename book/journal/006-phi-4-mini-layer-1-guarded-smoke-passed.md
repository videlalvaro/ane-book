---
layout: default
title: "Journal 006 - Phi-4-mini Layer-1 Guarded Smoke Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="005-phi-4-mini-layer-1-guarded-smoke-intent.html">Previous: Journal 005</a> | <a href="007-phi-4-mini-layers-2-3-bounded-smoke-intent.html">Next: Journal 007</a></nav>

# 2026-04-27 - Phi-4-mini Layer-1 Guarded Smoke Passed

**Intent**: Validate the generalized per-layer smoke gate on Phi-4-mini layer 1 before scale-out, following the quality-before-performance workflow and validation discipline.

**Setup**: Generalized the per-layer smoke gate; ran layer 1 only through guarded INT8 full-layer mlpackage conversion, compile, strict MLComputePlan residency, and numerical smoke. temporary JSON outputs.

**Result**: PASS. Layer 1 INT8 full-layer mlpackage conversion succeeded; compile succeeded; mlpackage/mlmodelc size was 96M. Strict residency passed: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Numerical smoke passed: cos(hidden)=0.999927, rmse=0.006605, max_abs=0.046875.

**Surprise / hurdle**: The generalized per-layer gate preserved layer-0 guardrails for a nonzero layer without ANE fallback.

**Lesson**: Phi-4-mini layer-local INT8 full-layer conversion is repeatably ANE-resident and numerically close across at least layers 0 and 1.

**Next**: No full conversion, perf, energy, or cleanup was run; continue only through gated scale-out after additional validation as needed.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="005-phi-4-mini-layer-1-guarded-smoke-intent.html">Previous: Journal 005</a> | <a href="007-phi-4-mini-layers-2-3-bounded-smoke-intent.html">Next: Journal 007</a></nav>
