---
layout: default
title: "Journal 030 - Phi-4-mini Three-Layer Full-Shard Probe Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="029-phi-4-mini-decode-profile-and-two-layer-probe-outcome.html">Previous: Journal 029</a> | <a href="031-phi-4-mini-three-layer-full-shard-probe-passed.html">Next: Journal 031</a></nav>

# 2026-04-27 - Phi-4-mini Three-Layer Full-Shard Probe Intent

**Intent**: Test a non-destructive 3-layer Phi-4-mini full INT8 stateful shard for layers 0–3 after the 2-layer probe passed, applying Dragon Book call-hoisting/strength reduction and Iverson whole-operation fusion.

**Setup**: Planned separate output directory: local artifacts; build and compile the layers 0–3 shard, then run strict MLComputePlan residency and multi-layer golden quality only if compile succeeds. No deletion/cleanup and no perf or energy benchmarking.

**Result**: Intent recorded before implementation; no 3-layer artifacts, compiled size, residency, quality, latency, or energy numbers yet.

**Surprise / hurdle**: The expected benefit is reducing layer CoreML calls from 32 to about 11, but compiled size may exceed the empirical ~250 MB ANE shard limit.

**Lesson**: Layer fusion should advance one larger shard at a time, with compiled-size, ANE residency, and golden quality gates before any performance claims.

**Next**: Build the separate 3-layer probe, compile it, then run strict residency and multi-layer golden quality if compilation stays under the limit.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="029-phi-4-mini-decode-profile-and-two-layer-probe-outcome.html">Previous: Journal 029</a> | <a href="031-phi-4-mini-three-layer-full-shard-probe-passed.html">Next: Journal 031</a></nav>
