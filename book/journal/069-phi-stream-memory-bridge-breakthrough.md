---
layout: default
title: "Journal 069 - Phi Stream-Level Memory Bridge Breakthrough"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="068-phi-stream-second-operation-boundary-narrowed.html">Previous: Journal 068</a> | <a href="070-phi-full-fused-topology-runs-in-one-stream.html">Next: Journal 070</a></nav>

# 2026-04-28 - Phi Stream-Level Memory Bridge Breakthrough

**Intent**: Determine why real Phi stage B still read zeros even though stage B `x` reported direct binding, state/inout memory objects were bound, and toy stateful chains passed.

**Setup**: Added stream runtime port-memory diagnostics to `local two-stage stream probe`: retain each port's memory object, query memory size, and query data pointer. Compared successful FP16 4D stateful toy chaining against failing Phi `23_24 -> 24_25`, then added an explicit `--manual-bind-memory` experiment using `manual memory binding` before manual encode.

**Result**: BREAKTHROUGH. The toy chain had different memory-object wrappers but the same producer/consumer `dataPtr`, so stage B consumed stage A output. Phi's stage B `x` had a different `dataPtr` despite `boundFeatureDirectly=YES`, explaining the zero output. Forcing the manual memory object bind changed stage B `x` to stage A's `dataPtr` and made manual Phi match public CoreML: one-layer `23_24 -> 24_25` stage B sum `-222.598015` in both paths. Every adjacent fused boundary in the best public topology now matches too: `0_16 -> 16_24` sum `4590.85129`, `16_24 -> 24_30` sum `-166.729431`, and `24_30 -> 30_32` sum `-116749.305`.

**Surprise / hurdle**: CoreML's direct binder state can report success while the underlying stream runtime input port still owns an independent buffer for large stateful Phi programs. The correct validation layer was the stream runtime memory object's data pointer, not binder metadata.

**Lesson**: The hidden-to-x edge must be expressed as a stream-level memory-object bind after CoreML input/state binding and before stream encode. Events and state/inout ports were distractions for this specific failure.

**Next**: Generalize the two-op probe to an N-op unsupported stream path for the full `16+8+6+2` topology, then profile latency/energy against the public CoreML runtime. Keep public CoreML as the correctness reference.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="068-phi-stream-second-operation-boundary-narrowed.html">Previous: Journal 068</a> | <a href="070-phi-full-fused-topology-runs-in-one-stream.html">Next: Journal 070</a></nav>
