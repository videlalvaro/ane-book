---
layout: default
title: "Journal 111 - O2: Concurrent FFN Partial Fan-Out"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="110-t4-1-5-closed-full-16-token-decode-exact-match-on-all-fp16-ane-stack.html">Previous: Journal 110</a> | <a href="112-int4-palettize-l0-ffn-probe-all-gates-pass.html">Next: Journal 112</a></nav>

# 2026-05-14 - O2: Concurrent FFN Partial Fan-Out

**Intent**: The 7 FFN partial shards per layer (p0–p6) are independent — same input `x`, additive `partial_moe` outputs. Prior implementation dispatched them sequentially in a for-loop. Replaced with `DispatchGroup` + concurrent `DispatchQueue` fan-out in local artifacts. Pre-allocated a stable `MLMultiArray` scratch buffer `[nPartials × dModel, Float16]` with non-overlapping row writes (row offset = `pi * dModel`) to avoid data races. After `group.wait()`: reduce scratch rows into `moeAccumF32`, then run `ffnLastModels` sequentially (depends on full sum). Motivation: eliminate the dominant per-layer latency for the 7 independent additions before the final combiner step, with zero correctness risk from the non-overlapping layout. Optimization discipline reference: the validation-first notes — measure parallelism headroom before introducing synchronisation overhead.

**Setup**: Hardware: M4 Max. Binary compiled as `/temporary output` (critical: must reuse the existing 32 GB ANE compilation cache — a new binary name forces a fresh 32 GB cache build that fills the local SSD). Runtime: all-FP16 ANE stack, 270 shards. Language: Swift. Key invariant: `nPartials = 7`, row stride = `dModel` (Float16), non-overlapping by construction.

**Result**: Binary compiles cleanly as `/temporary output`. Correctness verification pending — disk-full issues during decode runs blocked clean output capture. Whether ANE actually schedules concurrent `MLModel.prediction()` calls in parallel is an open research question; the implementation is correct regardless of the ANE scheduler's behaviour.

**Surprise / hurdle**: First compile attempt used binary name `/temporary output` → triggered a separate 32 GB CoreML cache under the CoreML cache for that binary → disk full at shard 10/270 → `[MIL FileWriter]` errors cascaded, blocking the entire decode run. Discovery cost: approximately 1 hour of compile time and a manual cache cleanup. The 32 GB-per-binary-name behaviour of the ANE compilation cache is non-obvious and undocumented.

**Lesson**: Always compile experimental Swift binaries as `/temporary output`; any new binary name triggers a separate 32 GB ANE cache rebuild that silently fills the local SSD.

**Next**: Confirm correctness (cosine vs FP16 reference) once disk headroom is cleared. If ANE does not internally parallelise concurrent `MLModel.prediction()` calls, the next step is to measure wall-clock delta vs sequential baseline to quantify the actual scheduling gain.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="110-t4-1-5-closed-full-16-token-decode-exact-match-on-all-fp16-ane-stack.html">Previous: Journal 110</a> | <a href="112-int4-palettize-l0-ffn-probe-all-gates-pass.html">Next: Journal 112</a></nav>
