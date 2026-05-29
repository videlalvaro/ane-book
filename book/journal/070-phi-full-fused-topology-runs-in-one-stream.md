---
layout: default
title: "Journal 070 - Phi Full Fused Topology Runs in One Stream-Level Path"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="069-phi-stream-memory-bridge-breakthrough.html">Previous: Journal 069</a> | <a href="071-phi-stream-one-stream-timing-reality-check.html">Next: Journal 071</a></nav>

# 2026-04-28 - Phi Full Fused Topology Runs in One Stream-Level Path

**Intent**: Move from pairwise unsupported stream path correctness to an N-op unsupported stream for the whole best public Phi topology, `16+8+6+2`.

**Setup**: Added `--manual-chain-all` to `local two-stage stream probe`, accepting multiple positional `.mlmodelc` paths. The path loads every operation into one CoreML stream execution stream, binds normal inputs/state for each operation, manually binds `stageN.hidden` memory into `stageN+1.x`, manually prepares and encodes all operations, then executes the stream once. Generalized `phi_public_two_call_probe` so the public reference can run the same ordered shard list in a separate process.

**Result**: PASS. The unsupported one-stream output matches public sequential CoreML at every stage for `phi4mini_layer0_16_q8 -> phi4mini_layer16_24_q8 -> phi4mini_layer24_30_q8 -> phi4mini_layer30_32_q8`: sums `4412.64955`, `4590.85129`, `4822.46835`, and final `-196.834778` in both paths.

**Surprise / hurdle**: Once the memory bridge was explicit, no extra event/sync-point work was needed for the four-op fused stack correctness probe.

**Lesson**: The public CoreML layer-shard roundtrip can be bypassed for Phi fused layer shards by constructing one stream-level stream and wiring hidden edges with stream-level memory-object binds.

**Next**: Turn the validated probe into a profiled decode path, then compare latency/energy against the current public `17.143 tok/s` runtime before deciding whether the unsupported path is worth productizing.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="069-phi-stream-memory-bridge-breakthrough.html">Previous: Journal 069</a> | <a href="071-phi-stream-one-stream-timing-reality-check.html">Next: Journal 071</a></nav>
