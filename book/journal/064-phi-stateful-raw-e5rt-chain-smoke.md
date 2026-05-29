---
layout: default
title: "Journal 064 - Phi Stateful Raw E5RT Chain Smoke"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="063-raw-e5rt-two-model-chain-breakthrough.html">Previous: Journal 063</a> | <a href="065-phi-public-two-call-reference-and-e5-event-probe.html">Next: Journal 065</a></nav>

# 2026-04-28 - Phi Stateful Raw E5RT Chain Smoke

**Intent**: Extend the raw E5RT two-operation stream breakthrough from toy controls to real Phi stateful shards, following call-hoisting/strength-reduction discipline and the [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md) stream-level execution focus.

**Setup**: Extended `e5_two_op_stream_probe` with `--phi-input` for real Phi shapes and state handling. Target chain: local artifacts -> local artifacts. Direct `MLState` did not work; `MLFeatureValue.internalFeatureValueWithState` requires a one-buffer `MLState`, so the probe uses `MLState.backings` and `MLState initWithBackings` per state port, then wraps each state with `internalFeatureValueWithState`. Stage B uses a provider overriding `x` with stage A hidden while CoreML's `_bindInputFeaturesAndWaitEvents` binds ordinary inputs and state.

**Result**: Both operation binders returned YES, both raw `e5rt_execution_stream_encode_operation` calls returned 0, and `_executeStream` returned YES. Stage A output was nonzero, with sample `[1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]` and sum `-337.912079`. Stage B output is currently all zeros.

**Surprise / hurdle**: Public prediction in the same process after taking private operations segfaulted, so the public reference path must run in a separate process. Stateful Phi ports also required one-buffer `MLState` wrappers rather than direct state reuse.

**Lesson**: Real stateful Phi two-shard E5 stream wiring and encoding works, but output correctness is not yet proven.

**Next**: Build a separate-process public reference and investigate output backing/state backing behavior for the all-zero stage B result before any performance claim or scale-out.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="063-raw-e5rt-two-model-chain-breakthrough.html">Previous: Journal 063</a> | <a href="065-phi-public-two-call-reference-and-e5-event-probe.html">Next: Journal 065</a></nav>
