---
layout: default
title: "Journal 063 - Raw E5RT Two-Model Chain Breakthrough"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="062-e5-setupoperationforinputfeatures-replaces-pool.html">Previous: Journal 062</a> | <a href="064-phi-stateful-raw-e5rt-chain-smoke.html">Next: Journal 064</a></nav>

# 2026-04-28 - Raw E5RT Two-Model Chain Breakthrough

**Intent**: Prove true cross-model chaining inside one E5 execution stream by using raw E5RT encode hooks instead of public host roundtrips, following the validation-first notes measurement/validation discipline and the [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md) focus on stream-level ANE behavior.

**Setup**: CoreML imports raw E5RT symbols from Espresso, including `e5rt_execution_stream_operation_prepare_op_for_encode` and `e5rt_execution_stream_encode_operation`. `e5_two_op_stream_probe` now `dlsym`s those symbols. Successful sequence: bind stage A input/output via ObjC operation private methods; bind stage A output feature/memory object into stage B input binder; bind stage B output; call raw E5RT prepare+encode for stage A and stage B operation handles into one stream; then call `MLE5ExecutionStream _executeStream:error`.

**Result**: PASS on the tiny distinct-input control `toy_a(x+1) -> toy_b_h(h*2)` with no `h` provider. Stage A hidden was `[2, 3, 4, 5]`; stage B hidden was `[4, 6, 8, 10]`. This proves stage B consumed stage A's output buffer and is the first true two-model E5 chain in this repo.

**Surprise / hurdle**: The key unlock was discovering that the public CoreML/Espresso stack already imports the raw E5RT prepare/encode symbols, making it possible to encode two operation handles into one stream while wiring the intermediate through private binders.

**Lesson**: Raw E5RT encode access turns cross-model chaining from a host-roundtrip problem into a stream construction problem.

**Next**: Validate the same path on two Phi layer-range shards against the public host-roundtrip output, then profile hidden-state copy removal.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="062-e5-setupoperationforinputfeatures-replaces-pool.html">Previous: Journal 062</a> | <a href="064-phi-stateful-raw-e5rt-chain-smoke.html">Next: Journal 064</a></nav>
