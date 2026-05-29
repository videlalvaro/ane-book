---
layout: default
title: "Journal 013 - Phi-4-mini Four-Layer Batch Runner Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="012-phi-4-mini-layers-4-7-bounded-run-range-passed.html">Previous: Journal 012</a> | <a href="014-phi-4-mini-layers-8-11-run-batches-intent.html">Next: Journal 014</a></nav>

# 2026-04-27 - Phi-4-mini Four-Layer Batch Runner Intent

**Intent**: Add a future-facing Phi-4-mini batched build runner that advances only bounded four-layer batches, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Planned orchestration: process one explicit four-layer batch at a time, check resources between batches, and reuse existing per-layer gates: convert, compile, strict MLComputePlan residency, and numerical smoke.

**Result**: Intent recorded before implementation; no artifacts, placement numbers, latency, energy, cosine, or perplexity results yet.

**Surprise / hurdle**: The runner must make batched progress convenient without silently running all 32 layers or bypassing per-layer gates.

**Lesson**: Batch automation is safe only when batch size is bounded, resource checks happen between batches, and every layer still passes the same gates.

**Next**: Implement the runner non-destructively; do not run performance or energy tests, do not clean up/delete artifacts, and require explicit user action for any full 32-layer scale-out.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="012-phi-4-mini-layers-4-7-bounded-run-range-passed.html">Previous: Journal 012</a> | <a href="014-phi-4-mini-layers-8-11-run-batches-intent.html">Next: Journal 014</a></nav>
