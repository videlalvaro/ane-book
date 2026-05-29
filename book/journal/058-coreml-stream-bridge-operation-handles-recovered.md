---
layout: default
title: "Journal 058 - CoreML Stream-Level Bridge Operation Handles Recovered"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="057-stream-dispatch-bridge-outcome.html">Previous: Journal 057</a> | <a href="059-coreml-stream-two-operation-binder-outcome.html">Next: Journal 059</a></nav>

# 2026-04-28 - CoreML Stream-Level Bridge Operation Handles Recovered

**Intent**: Advance the unsupported CoreML stream-level investigation from program-library discovery toward operation-level chaining, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline to remove host materialization between validated shards.

**Setup**: Added local inspection artifacts to map the CoreML stream lifecycle for Phi layer30_32 and layer16_24 shards.

**Result**: The inspection recovered enough operation and port identity to reason about `x` input, `hidden` output, masks, and KV state on both tested shards.

**Surprise / hurdle**: The useful bridge surface is partly CoreML (CoreML stream operation pool / CoreML stream execution operation) and partly stream-level pointers, so object introspection alone misses the operation contract.

**Lesson**: Public CoreML-loaded stream-level models expose enough live operation and port handles to make ANE-side shard binding a concrete next experiment.

**Next**: Construct or borrow an CoreML stream execution stream containing two operations and bind stage A `hidden` output directly to stage B `x` input without `MLMultiArray` host materialization; keep public residency and golden gates as acceptance checks before any performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="057-stream-dispatch-bridge-outcome.html">Previous: Journal 057</a> | <a href="059-coreml-stream-two-operation-binder-outcome.html">Next: Journal 059</a></nav>
