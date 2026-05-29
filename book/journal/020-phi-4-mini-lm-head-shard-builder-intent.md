---
layout: default
title: "Journal 020 - Phi-4-mini LM Head Shard Builder Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="019-phi-4-mini-all-32-layer-shards-completed.html">Previous: Journal 019</a> | <a href="021-phi-4-mini-lm-head-shards-passed.html">Next: Journal 021</a></nav>

# 2026-04-27 - Phi-4-mini LM Head Shard Builder Intent

**Intent**: After all 32 Phi-4-mini layer shards completed and the user said “keep going from the top,” proceed to the next ANE-only compute-heavy component: final RMSNorm plus LM head projection, following validation-before-scale discipline and the project ANE-only mandate.

**Setup**: Planned implementation: build a Phi-4-mini LM head shard builder that reads `token_embd.weight` as the tied LM head and `output_norm.weight` from GGUF, splits vocab=200064 into 4 INT8 CoreML shards, and emits one RMSNorm+Conv2d shard per vocab slice for Xcode Python/CoreML compilation.

**Result**: Intent recorded before implementation; no LM-head artifacts, placement numbers, latency, energy, cosine, or perplexity results yet.

**Surprise / hurdle**: The host-side LM head remains compute-heavy and must not be optimized as a CPU/GPU shortcut; shard 0 must prove ANE residency before scaling to the other vocab shards.

**Lesson**: Once transformer layers are ANE-resident, the final projection becomes the next mandatory ANE shard rather than an optional runtime optimization.

**Next**: Implement the builder, compile and validate shard 0 residency first, then build and validate shards 1–3 only if shard 0 passes; do not run perf/energy benchmarking and do not clean up or delete artifacts.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="019-phi-4-mini-all-32-layer-shards-completed.html">Previous: Journal 019</a> | <a href="021-phi-4-mini-lm-head-shards-passed.html">Next: Journal 021</a></nav>
