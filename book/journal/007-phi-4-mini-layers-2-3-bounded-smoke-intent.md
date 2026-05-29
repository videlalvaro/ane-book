---
layout: default
title: "Journal 007 - Phi-4-mini Layers 2–3 Bounded Smoke Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="006-phi-4-mini-layer-1-guarded-smoke-passed.html">Previous: Journal 006</a> | <a href="008-phi-4-mini-layers-2-3-bounded-smoke-passed.html">Next: Journal 008</a></nav>

# 2026-04-27 - Phi-4-mini Layers 2–3 Bounded Smoke Intent

**Intent**: Continue Phi-4-mini-instruct ANE support with a bounded batch of layers 2 and 3 only, using the established guardrails and validation-before-scale discipline.

**Setup**: Planned flow: guarded preflight → convert → compile → strict MLComputePlan residency → per-layer PyTorch-vs-CoreML numerical smoke for layers 2 and 3. Constraints: no full 32-layer conversion, no perf/energy run, and no cleanup or deletion.

**Result**: Intent recorded before execution; no new placement, latency, energy, cosine, perplexity, or artifact counts yet.

**Surprise / hurdle**: The immediate risk is keeping a two-layer batch bounded so it exercises repeatability without becoming scale-out.

**Lesson**: Small bounded batches can test process repeatability while preserving the ANE residency and quality gates before any expensive expansion.

**Next**: Run only layers 2 and 3 through the guarded flow; record residency and numerical-smoke results in a follow-up entry.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="006-phi-4-mini-layer-1-guarded-smoke-passed.html">Previous: Journal 006</a> | <a href="008-phi-4-mini-layers-2-3-bounded-smoke-passed.html">Next: Journal 008</a></nav>
