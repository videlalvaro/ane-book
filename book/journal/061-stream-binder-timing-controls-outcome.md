---
layout: default
title: "Journal 061 - Stream-Level Binder Timing Controls Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="060-tiny-stream-execution-controls-outcome.html">Previous: Journal 060</a> | <a href="062-stream-setupoperationforinputfeatures-replaces-pool.html">Next: Journal 062</a></nav>

# 2026-04-28 - Stream-Level Binder Timing Controls Outcome

**Intent**: Pin down when unsupported CoreML Stream-Level port bindings become fixed, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline for removing shard-boundary materialization without relying on post-hoc host mutation.

**Setup**: Updated local artifacts to test when CoreML stream input and output bindings become fixed on toy stream-level control models.

**Result**: Stage B kept using its explicit provider input rather than the attempted stage-A output binding. After stream preparation, CoreML rejected binding changes because the operation was already in use by the execution stream. Running without proper stream preparation failed because no operations had been encoded.

**Surprise / hurdle**: The binder API can accept memory objects and feature values, but stream preparation encodes the operations and locks the binding plan before any later MLFeatureValue or binder mutation can express a cross-model edge.

**Lesson**: True stream-level cross-model chaining must be expressed before or inside `setupOperationForInputFeatures` or lower stream runtime setup; post-prepare MLFeatureValue/binder mutation is too late.

**Next**: Search below the prepared stream boundary for an stream runtime setup or port-link primitive that can connect output and input ports before encoding; do not apply the post-hoc binder path to Phi shards.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="060-tiny-stream-execution-controls-outcome.html">Previous: Journal 060</a> | <a href="062-stream-setupoperationforinputfeatures-replaces-pool.html">Next: Journal 062</a></nav>
