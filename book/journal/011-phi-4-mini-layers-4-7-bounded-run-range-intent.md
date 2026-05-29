---
layout: default
title: "Journal 011 - Phi-4-mini Layers 4–7 Bounded Run-Range Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="010-phi-4-mini-bounded-run-range-orchestration-landed.html">Previous: Journal 010</a> | <a href="012-phi-4-mini-layers-4-7-bounded-run-range-passed.html">Next: Journal 012</a></nav>

# 2026-04-27 - Phi-4-mini Layers 4–7 Bounded Run-Range Intent

**Intent**: Start the first actual bounded Phi-4-mini build+test using the new `run-range` stage for layers 4 through 7 only, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Planned command shape: `the Phi orchestration script run-range --layer-start 4 --layer-end 8 --gatekeeper-go`; per-layer flow: preflight before each layer, convert, compile, strict MLComputePlan residency, and numerical smoke.

**Result**: Intent recorded before execution; no placement, cosine, latency, energy, or artifact-count results yet.

**Surprise / hurdle**: The key risk is proving the range runner can execute real work for four layers while staying bounded and re-checking resources before each layer.

**Lesson**: A small real range is the next safe step after dry-run orchestration, but it must not become full 32-layer scale-out.

**Next**: Run only layers 4–7 through the guarded flow; do not run perf/energy and do not clean up or delete artifacts.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="010-phi-4-mini-bounded-run-range-orchestration-landed.html">Previous: Journal 010</a> | <a href="012-phi-4-mini-layers-4-7-bounded-run-range-passed.html">Next: Journal 012</a></nav>
