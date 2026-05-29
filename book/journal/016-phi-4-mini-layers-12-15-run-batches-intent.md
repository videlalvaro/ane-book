---
layout: default
title: "Journal 016 - Phi-4-mini Layers 12–15 Run-Batches Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="015-phi-4-mini-run-batches-landed-and-layers-8-11-passed.html">Previous: Journal 015</a> | <a href="017-phi-4-mini-layers-12-15-run-batches-passed.html">Next: Journal 017</a></nav>

# 2026-04-27 - Phi-4-mini Layers 12–15 Run-Batches Intent

**Intent**: Continue the actual Phi-4-mini bounded build+test with `run-batches` for layers 12–15 only, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Planned command: the bounded batch stage of the Phi orchestration script; completion proceeds in explicit four-layer batches with preflight, convert, compile, strict MLComputePlan residency, and numerical smoke gates.

**Result**: Intent recorded before execution; no placement, cosine, latency, energy, or artifact-count results yet.

**Surprise / hurdle**: The key risk is ensuring this remains one checked bounded batch rather than an unchecked full 32-layer conversion.

**Lesson**: Four-layer batches keep scale-out auditable when every batch is explicit and every layer must pass residency plus numerical smoke gates.

**Next**: Run only layers 12–15 through the guarded batch; do not run performance or energy tests, and do not clean up or delete artifacts.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="015-phi-4-mini-run-batches-landed-and-layers-8-11-passed.html">Previous: Journal 015</a> | <a href="017-phi-4-mini-layers-12-15-run-batches-passed.html">Next: Journal 017</a></nav>
