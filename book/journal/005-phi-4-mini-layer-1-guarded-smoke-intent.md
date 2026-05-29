---
layout: default
title: "Journal 005 - Phi-4-mini Layer-1 Guarded Smoke Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="004-phi-4-mini-layer-0-numerical-smoke-passed.html">Previous: Journal 004</a> | <a href="006-phi-4-mini-layer-1-guarded-smoke-passed.html">Next: Journal 006</a></nav>

# 2026-04-27 - Phi-4-mini Layer-1 Guarded Smoke Intent

**Intent**: Continue Phi-4-mini-instruct ANE support by generalizing the per-layer PyTorch-vs-CoreML smoke gate beyond layer 0, then exercising only layer 1 through the same guarded path, following the project quality-before-scale workflow and validation discipline.

**Setup**: Planned scope: parameterize the existing layer smoke gate for nonzero layers; run layer 1 only through guarded conversion, compile, strict residency validation, and numerical smoke. Constraints: no full conversion, no perf or energy run, and no cleanup/deletion.

**Result**: Intent recorded before implementation; no new artifacts, placement numbers, latency, energy, cosine, or perplexity yet.

**Surprise / hurdle**: The next risk is whether layer-index generalization preserves the layer-0 guardrails without accidentally triggering scale-out work.

**Lesson**: Scale-out should advance one representative layer at a time until conversion, residency, and numerical-smoke invariants are repeatable.

**Next**: Implement the parameterized gate and run only layer 1; record residency and numerical-smoke results in a follow-up entry.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="004-phi-4-mini-layer-0-numerical-smoke-passed.html">Previous: Journal 004</a> | <a href="006-phi-4-mini-layer-1-guarded-smoke-passed.html">Next: Journal 006</a></nav>
