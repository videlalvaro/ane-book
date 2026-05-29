---
layout: default
title: "Journal 017 - Phi-4-mini Layers 12–15 Run-Batches Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="016-phi-4-mini-layers-12-15-run-batches-intent.html">Previous: Journal 016</a> | <a href="018-phi-4-mini-remaining-layers-16-31-run-batches-intent.html">Next: Journal 018</a></nav>

# 2026-04-27 - Phi-4-mini Layers 12–15 Run-Batches Passed

**Intent**: Record the next explicit Phi-4-mini guarded four-layer batch outcome, continuing validation-before-scale per the validation-first notes and the project ANE-only gate policy.

**Setup**: Ran the bounded batch stage of the Phi orchestration script; one explicit guarded batch for layers 12–15 with convert, compile, strict MLComputePlan residency, and numerical smoke gates.

**Result**: PASS. Layers 12, 13, 14, and 15 converted and compiled successfully. For each layer, mlpackage=96M, mlmodelc=96M, and meta=4.0K. Strict MLComputePlan residency passed for each layer: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Numerical smoke passed: L12 cos=0.999841 rmse=0.016735 max_abs=0.061035; L13 cos=0.999818 rmse=0.017346 max_abs=0.063477; L14 cos=0.999833 rmse=0.018678 max_abs=0.088867; L15 cos=0.999834 rmse=0.018093 max_abs=0.078125.

**Surprise / hurdle**: Scale-out is proceeding toward all layers only through explicit guarded four-layer batches, not as an unchecked full conversion.

**Lesson**: Phi-4-mini INT8 full-layer shards remain ANE-resident and numerically close through layer 15 when advanced by explicit guarded batches.

**Next**: No performance run, energy run, cleanup, or deletion was run; continue only through explicit guarded four-layer batches.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="016-phi-4-mini-layers-12-15-run-batches-intent.html">Previous: Journal 016</a> | <a href="018-phi-4-mini-remaining-layers-16-31-run-batches-intent.html">Next: Journal 018</a></nav>
