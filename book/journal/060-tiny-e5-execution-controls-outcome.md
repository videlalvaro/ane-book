---
layout: default
title: "Journal 060 - Tiny E5 Execution Controls Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="059-coreml-e5-two-operation-stream-binder-outcome.html">Previous: Journal 059</a> | <a href="061-e5-binder-timing-controls-outcome.html">Next: Journal 061</a></nav>

# 2026-04-28 - Tiny E5 Execution Controls Outcome

**Intent**: Test minimal E5 execution control models before full Phi chaining, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline for removing shard-boundary materialization.

**Setup**: Generated workspace-local toy CoreML models: `toy_a` computes `x+1`, `toy_b` computes `x*2`, and `toy_b_h` computes `h*2`. Ran a two-operation `MLE5ExecutionStream` with `toy_a+toy_b`, then a distinct-input `toy_a+toy_b_h` test with forced direct binder state. Broad dyld extraction was deferred because disk free was about 25 GiB.

**Result**: The `toy_a+toy_b` two-op stream executed successfully. Stage A hidden was `[2,3,4,5]`; stage B hidden was `[2,4,6,8]`, proving the stream can execute two operations but that B consumed original `x` rather than A `hidden`. The `toy_a+toy_b_h` distinct-input test failed `executeForInputFeatures` with `The input feature is invalid or unsupported. (port trait Tensor, feature trait Unknown.)` despite forced binder direct state.

**Surprise / hurdle**: MLFeatureValue-level reuse and forced direct binder state were not enough to express an output-to-input edge between operations.

**Lesson**: Two-op E5 streams can run, but hidden-to-input chaining needs a lower E5RT/e5rt_io_port output-to-input link primitive rather than MLFeatureValue reuse.

**Next**: Search for the lower E5RT port-linking primitive before applying E5 chaining to Phi shards; keep broad dyld extraction deferred until disk headroom improves.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="059-coreml-e5-two-operation-stream-binder-outcome.html">Previous: Journal 059</a> | <a href="061-e5-binder-timing-controls-outcome.html">Next: Journal 061</a></nav>
