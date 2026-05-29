---
layout: default
title: "Journal 061 - E5 Binder Timing Controls Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="060-tiny-e5-execution-controls-outcome.html">Previous: Journal 060</a> | <a href="062-e5-setupoperationforinputfeatures-replaces-pool.html">Next: Journal 062</a></nav>

# 2026-04-28 - E5 Binder Timing Controls Outcome

**Intent**: Pin down when private CoreML E5 port bindings become fixed, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline for removing shard-boundary materialization without relying on post-hoc host mutation.

**Setup**: Updated local artifacts to call `MLE5InputPortBinder bindMemoryObjectForFeatureValue`, operation-level `_bindInputFeaturesAndWaitEvents` / `_bindOutputPortsWithOptions`, stream `_prepareForInputFeatures`, and raw `_executeStream` on toy E5 control models.

**Result**: `bindMemoryObjectForFeatureValue` returned YES with the stage A output feature, but did not create a true chain. With `toy_b_h` and an explicit `h` provider `[10,20,30,40]`, stage B output was `[20,40,60,80]`, proving the provider input wins over the attempted A-output binding. After stream `_prepareForInputFeatures`, attempts to change port bindings failed with `Port bindings cannot be changed while operation is in use in an execution stream.` Raw `_executeStream` after prepare worked but used provider `h`; raw `_executeStream` without stream preparation failed with `No operations have been encoded to the execution stream.`

**Surprise / hurdle**: The binder API can accept memory objects and feature values, but stream preparation encodes the operations and locks the binding plan before any later MLFeatureValue or binder mutation can express a cross-model edge.

**Lesson**: True E5 cross-model chaining must be expressed before or inside `setupOperationForInputFeatures` or lower E5RT setup; post-prepare MLFeatureValue/binder mutation is too late.

**Next**: Search below the prepared stream boundary for an E5RT setup or port-link primitive that can connect output and input ports before encoding; do not apply the post-hoc binder path to Phi shards.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="060-tiny-e5-execution-controls-outcome.html">Previous: Journal 060</a> | <a href="062-e5-setupoperationforinputfeatures-replaces-pool.html">Next: Journal 062</a></nav>
