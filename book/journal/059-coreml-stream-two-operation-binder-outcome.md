---
layout: default
title: "Journal 059 - CoreML Stream-Level Two-Operation Stream Binder Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="058-coreml-stream-bridge-operation-handles-recovered.html">Previous: Journal 058</a> | <a href="060-tiny-stream-execution-controls-outcome.html">Next: Journal 060</a></nav>

# 2026-04-28 - CoreML Stream-Level Two-Operation Stream Binder Outcome

**Intent**: Test whether two already-loaded unsupported CoreML Stream-Level operations from adjacent Phi shards can be placed into one CoreML stream execution stream, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline to reduce shard-boundary materialization.

**Setup**: Added local artifacts; loaded adjacent Phi shards 16_24 then 24_30; extracted one CoreML stream execution operation from each; constructed a single CoreML stream execution stream containing both operations; probed stage A `hidden` output binding and stage B `x` input binding.

**Result**: `serializeInferenceFrameDataForOptions:error` returned YES for the two-operation stream. Stage A `hidden` is a directly bound output. Stage B `x` is not direct by default. Stage A `hidden prepareWithOptions` yields an `MLFeatureValue` MultiArray; stage B `x` accepts it through `prepareForFeatureValue`, but `xDirect` remains NO. `CoreML streamInputPortBinder _reusableForFeatureValue:directMode` reports mode 2 -> YES; forcing `setDirectlyBoundFeatureValue` plus `setBindingMode:1` makes stage B `x boundFeatureDirectly` YES.

**Surprise / hurdle**: The hidden-to-x bridge can be made structurally direct, but the default binder path does not automatically preserve direct binding across the two operations.

**Lesson**: Unsupported Stream-Level hidden-to-x direct binder state can be forced structurally, but correctness depends on executing a tiny two-model graph before applying it to full Phi.

**Next**: Run an execution test on a tiny two-model graph with the forced binder state before any full Phi chaining, performance claim, cleanup, deletion, or scale-out.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="058-coreml-stream-bridge-operation-handles-recovered.html">Previous: Journal 058</a> | <a href="060-tiny-stream-execution-controls-outcome.html">Next: Journal 060</a></nav>
