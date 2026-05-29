---
layout: default
title: "Journal 062 - E5 setupOperationForInputFeatures Replaces Pool"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="061-e5-binder-timing-controls-outcome.html">Previous: Journal 061</a> | <a href="063-raw-e5rt-two-model-chain-breakthrough.html">Next: Journal 063</a></nav>

# 2026-04-28 - E5 setupOperationForInputFeatures Replaces Pool

**Intent**: Determine whether the ObjC `MLE5ExecutionStream setupOperationForInputFeatures:operationPool:error:` surface can append multiple operations to one stream, following call-hoisting/strength-reduction discipline for reducing shard-boundary materialization.

**Setup**: Ran `e5_two_op_stream_probe --probe-setup` on a fresh stream. The probe called `setupOperationForInputFeatures:operationPool:error:` twice: first with the `toy_a` pool, then with the `toy_b` / `toy_b_h` pool.

**Result**: Both setup calls returned YES. After the first call, `operations` contained one op from `toy_a` and `operationPool` was `toy_a`; after the second call, `operations` contained one op from `toy_b` / `toy_b_h` and `operationPool` was the second pool. The second call replaced the stream contents rather than appending. `serializeInferenceFrameDataForOptions` returned YES, but raw `_executeStream` reported `No operations have been encoded to the execution stream.`

**Surprise / hurdle**: The public ObjC setup surface looks one-operation/one-pool oriented and does not expose a multi-op DAG or append encoder.

**Lesson**: `MLE5ExecutionStream setupOperationForInputFeatures` is not the missing append/chaining primitive; it replaces the active operation pool.

**Next**: Remaining paths are raw E5RT below the ObjC wrapper or building one CoreML program/function containing fused ranges; do not spend more Phi chaining work on this ObjC setup surface.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="061-e5-binder-timing-controls-outcome.html">Previous: Journal 061</a> | <a href="063-raw-e5rt-two-model-chain-breakthrough.html">Next: Journal 063</a></nav>
