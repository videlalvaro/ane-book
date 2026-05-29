---
layout: default
title: "Journal 070 - Phi Full Fused Topology Runs in One E5 Stream"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="069-phi-e5-raw-memory-bridge-breakthrough.html">Previous: Journal 069</a> | <a href="071-phi-private-e5-one-stream-timing-reality-check.html">Next: Journal 071</a></nav>

# 2026-04-28 - Phi Full Fused Topology Runs in One E5 Stream

**Intent**: Move from pairwise private E5 correctness to an N-op private stream for the whole best public Phi topology, `16+8+6+2`.

**Setup**: Added `--manual-chain-all` to `e5_two_op_stream_probe`, accepting multiple positional `.mlmodelc` paths. The path loads every operation into one `MLE5ExecutionStream`, binds normal inputs/state for each operation, raw-binds `stageN.hidden` memory into `stageN+1.x`, raw-prepares/encodes all operations, then executes the stream once. Generalized `phi_public_two_call_probe` so the public reference can run the same ordered shard list in a separate process.

**Result**: PASS. The private one-stream output matches public sequential CoreML at every stage for `phi4mini_layer0_16_q8 -> phi4mini_layer16_24_q8 -> phi4mini_layer24_30_q8 -> phi4mini_layer30_32_q8`: sums `4412.64955`, `4590.85129`, `4822.46835`, and final `-196.834778` in both paths.

**Surprise / hurdle**: Once the raw memory bridge was explicit, no extra event/sync-point work was needed for the four-op fused stack correctness probe.

**Lesson**: The public CoreML layer-shard roundtrip can be bypassed for Phi fused layer shards by constructing one E5 stream and wiring hidden edges with raw E5RT memory-object binds.

**Next**: Turn the validated probe into a profiled decode path, then compare latency/energy against the current public `17.143 tok/s` runtime before deciding whether the private path is worth productizing.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="069-phi-e5-raw-memory-bridge-breakthrough.html">Previous: Journal 069</a> | <a href="071-phi-private-e5-one-stream-timing-reality-check.html">Next: Journal 071</a></nav>
