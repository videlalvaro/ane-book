---
layout: default
title: "Journal 004 - Phi-4-mini Layer-0 Numerical Smoke Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="003-phi-4-mini-instruct-layer-0-gate-residency-passed.html">Previous: Journal 003</a> | <a href="005-phi-4-mini-layer-1-guarded-smoke-intent.html">Next: Journal 005</a></nav>

# 2026-04-27 - Phi-4-mini Layer-0 Numerical Smoke Passed

**Intent**: Add a cheap per-layer numerical smoke gate before scale-out, comparing PyTorch FP16 layer-0 hidden states against CoreML INT8 output under the quality-before-performance workflow.

**Setup**: Added stage `golden-layer` in the Phi orchestration script; PyTorch FP16 vs CoreML INT8 layer-0 smoke gate; temporary JSON output.

**Result**: PASS: cos(hidden)=0.999958, rmse=0.004737, max_abs=0.026367.

**Surprise / hurdle**: This validates only the layer-0 numerical smoke path; it is not a full-model golden validation.

**Lesson**: A lightweight per-layer golden smoke can catch obvious CoreML conversion drift before paying for full-model gates.

**Next**: Keep full-model golden validation as the required quality gate before benchmarking or shipping Phi-4-mini artifacts.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="003-phi-4-mini-instruct-layer-0-gate-residency-passed.html">Previous: Journal 003</a> | <a href="005-phi-4-mini-layer-1-guarded-smoke-intent.html">Next: Journal 005</a></nav>
