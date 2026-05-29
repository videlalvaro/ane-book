---
layout: default
title: "Journal 058 - CoreML E5 Bridge Operation Handles Recovered"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="057-private-ane-api-bridge-outcome.html">Previous: Journal 057</a> | <a href="059-coreml-e5-two-operation-stream-binder-outcome.html">Next: Journal 059</a></nav>

# 2026-04-28 - CoreML E5 Bridge Operation Handles Recovered

**Intent**: Advance the private ANE/CoreML E5 bridge from program-library discovery toward operation-level chaining, following the validation-first notes Dragon Book call-hoisting/strength-reduction discipline to remove host materialization between validated shards.

**Setup**: Added local artifacts to dump live CoreML E5 classes. Updated local artifacts to reach `MLE5StaticShapeExecutionStreamOperationPool`, call `prepareWithInitialPoolSize:error:`, `_takeOut` an `MLE5ExecutionStreamOperation`, and dump operation plus port handles on Phi layer30_32 and layer16_24 shards.

**Result**: Discovered `MLE5ProgramLibrary.createOperationForFunctionName` returns raw `e5rt_execution_stream_operation*`, not an ObjC object. Recovered `e5rt_program_library*`, `e5rt_execution_stream_operation*`, and named `e5rt_io_port*` handles for `x` input, `hidden` output, masks, and KV state on both tested shards.

**Surprise / hurdle**: The useful bridge surface is partly ObjC (`MLE5StaticShapeExecutionStreamOperationPool` / `MLE5ExecutionStreamOperation`) and partly raw E5RT pointers, so object introspection alone misses the operation contract.

**Lesson**: Public CoreML-loaded E5 models expose enough live operation and port handles to make ANE-side shard binding a concrete next experiment.

**Next**: Construct or borrow an `MLE5ExecutionStream` containing two operations and bind stage A `hidden` output directly to stage B `x` input without `MLMultiArray` host materialization; keep public residency and golden gates as acceptance checks before any performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="057-private-ane-api-bridge-outcome.html">Previous: Journal 057</a> | <a href="059-coreml-e5-two-operation-stream-binder-outcome.html">Next: Journal 059</a></nav>
