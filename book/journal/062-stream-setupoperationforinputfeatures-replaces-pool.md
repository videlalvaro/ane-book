---
layout: default
title: "Journal 062 - Stream-Level SetupOperationForInputFeatures Replaces Pool"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="061-stream-binder-timing-controls-outcome.html">Previous: Journal 061</a> | <a href="063-stream-two-model-chain-breakthrough.html">Next: Journal 063</a></nav>

# 2026-04-28 - Stream-Level SetupOperationForInputFeatures Replaces Pool

**Intent**: Determine whether the CoreML `CoreML stream execution stream setupOperationForInputFeatures:operationPool:error:` surface can append multiple operations to one stream, following call-hoisting/strength-reduction discipline for reducing shard-boundary materialization.

**Setup**: Ran `local two-stage stream probe --probe-setup` on a fresh stream. The probe called `setupOperationForInputFeatures:operationPool:error:` twice: first with the `toy_a` pool, then with the `toy_b` / `toy_b_h` pool.

**Result**: Both setup calls returned YES, but the second call replaced the stream contents rather than appending to them. The stream did not become a two-operation chain through this public-looking setup route.

**Surprise / hurdle**: The public CoreML setup surface looks one-operation/one-pool oriented and does not expose a multi-op DAG or append encoder.

**Lesson**: `CoreML stream execution stream setupOperationForInputFeatures` is not the missing append/chaining primitive; it replaces the active operation pool.

**Next**: Remaining paths are stream-level below the CoreML wrapper or building one CoreML program/function containing fused ranges; do not spend more Phi chaining work on this CoreML setup surface.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="061-stream-binder-timing-controls-outcome.html">Previous: Journal 061</a> | <a href="063-stream-two-model-chain-breakthrough.html">Next: Journal 063</a></nav>
