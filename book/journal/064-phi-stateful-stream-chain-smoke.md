---
layout: default
title: "Journal 064 - Phi Stateful Stream-Level Chain Smoke"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="063-stream-two-model-chain-breakthrough.html">Previous: Journal 063</a> | <a href="065-phi-public-two-call-reference-and-stream-event-probe.html">Next: Journal 065</a></nav>

# 2026-04-28 - Phi Stateful Stream-Level Chain Smoke

**Intent**: Extend the stream-level two-operation stream breakthrough from toy controls to real Phi stateful shards, following call-hoisting/strength-reduction discipline and the [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md) stream-level execution focus.

**Setup**: Extended the local two-stage stream probe with `--phi-input` for real Phi shapes and state handling. Target chain: local artifacts -> local artifacts. Stage B used a provider overriding `x` with stage A hidden while CoreML handled ordinary inputs and state.

**Result**: The stream-level encode path completed. Stage A output was nonzero, with sample `[1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]` and sum `-337.912079`. Stage B output is currently all zeros.

**Surprise / hurdle**: Public prediction in the same process after taking unsupported operations segfaulted, so the public reference path must run in a separate process. Stateful Phi ports also required one-buffer `MLState` wrappers rather than direct state reuse.

**Lesson**: Real stateful Phi two-shard stream-level stream wiring and encoding works, but output correctness is not yet proven.

**Next**: Build a separate-process public reference and investigate output backing/state backing behavior for the all-zero stage B result before any performance claim or scale-out.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="063-stream-two-model-chain-breakthrough.html">Previous: Journal 063</a> | <a href="065-phi-public-two-call-reference-and-stream-event-probe.html">Next: Journal 065</a></nav>
