---
layout: default
title: "Journal 014 - Phi-4-mini Layers 8–11 Run-Batches Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="013-phi-4-mini-four-layer-batch-runner-intent.html">Previous: Journal 013</a> | <a href="015-phi-4-mini-run-batches-landed-and-layers-8-11-passed.html">Next: Journal 015</a></nav>

# 2026-04-27 - Phi-4-mini Layers 8–11 Run-Batches Intent

**Intent**: Use the new `run-batches` stage for the next actual bounded Phi-4-mini build+test batch, layers 8–11 only, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Planned command: the bounded batch stage of the Phi orchestration script; preflight before the batch, then existing per-layer convert, compile, strict MLComputePlan residency, and numerical smoke gates.

**Result**: Intent recorded before execution; no placement, cosine, latency, energy, or artifact-count results yet.

**Surprise / hurdle**: The key risk is proving batch orchestration can run one real four-layer batch without expanding into full 32-layer conversion or bypassing per-layer gates.

**Lesson**: A stopped-after-one batch is the safe next step for batch automation when every layer still passes residency and numerical smoke before scale-out.

**Next**: Run only layers 8–11 through the guarded batch; do not run full 32-layer conversion, performance, energy, cleanup, or deletion.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="013-phi-4-mini-four-layer-batch-runner-intent.html">Previous: Journal 013</a> | <a href="015-phi-4-mini-run-batches-landed-and-layers-8-11-passed.html">Next: Journal 015</a></nav>
