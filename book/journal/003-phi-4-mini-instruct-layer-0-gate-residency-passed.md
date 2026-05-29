---
layout: default
title: "Journal 003 - Phi-4-mini-instruct Layer-0 Gate Residency Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="002-phi-4-mini-instruct-ane-support-scaffolding-intent.html">Previous: Journal 002</a> | <a href="004-phi-4-mini-layer-0-numerical-smoke-passed.html">Next: Journal 004</a></nav>

# 2026-04-27 - Phi-4-mini-instruct Layer-0 Gate Residency Passed

**Intent**: Validate the smallest representative Phi-4-mini-instruct full-layer INT8 CoreML shard before scale-out, following the ANE residency gate and optimization discipline from the validation-first notes.

**Setup**: Layer 0 full-layer INT8 mlpackage conversion, CoreML compilation to mlmodelc, strict MLComputePlan residency check.

**Result**: Conversion succeeded; compiled mlmodelc succeeded; package/modelc size was 96M. Strict residency passed: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0.

**Surprise / hurdle**: No fallback ops appeared in the strict plan for the representative layer-0 gate.

**Lesson**: Phi-4-mini-instruct layer-0 INT8 full-layer packaging is a viable ANE-resident pattern to consider for scale-out.

**Next**: Perf, energy, full conversion, and cleanup were not run; proceed only through the normal gated scale-out flow.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="002-phi-4-mini-instruct-ane-support-scaffolding-intent.html">Previous: Journal 002</a> | <a href="004-phi-4-mini-layer-0-numerical-smoke-passed.html">Next: Journal 004</a></nav>
