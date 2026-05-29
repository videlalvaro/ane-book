---
layout: default
title: "Journal 025 - Phi-4-mini Lean ANE Runtime Optimization Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="024-phi-4-mini-provisional-tok-s-timing-smoke.html">Previous: Journal 024</a> | <a href="026-phi-4-mini-lean-ane-runtime-optimization-outcome.html">Next: Journal 026</a></nav>

# 2026-04-27 - Phi-4-mini Lean ANE Runtime Optimization Intent

**Intent**: Start optimizing toward a leaner ANE model/runtime to save energy for coding agents by applying low-risk host-overhead changes before any energy benchmarking. The plan follows Iverson/APL whole-array primitive thinking by treating the four LM-head shards as independent array partitions, Dragon Book strength reduction/allocation hoisting by reusing CoreML input providers, and Stepanov's semigroup/reduction framing by reducing four local argmaxes into one global argmax.

**Setup**: Planned scope: Phi-4-mini Swift prompt-ID runtime only; reuse CoreML input providers instead of allocating a new MLDictionaryFeatureProvider for each layer/token; copy every layer output into the reusable x buffer; dispatch the 4 independent ANE LM-head shards concurrently; perform a four-way argmax reduction on host over shard-local results. Constraints: preserve ANE-only heavy compute, keep LM-head projection in CoreML ANE shards with no CPU fallback, run no powermetrics/energy benchmark yet, and perform no cleanup/deletion.

**Result**: Intent recorded before implementation; no placement, latency, energy, cosine, perplexity, or artifact numbers yet.

**Surprise / hurdle**: The optimization target is host overhead around already-ANE heavy compute, so correctness and ANE residency must remain unchanged while allocations and serial shard dispatch are reduced.

**Lesson**: Energy-oriented runtime work should first remove avoidable host allocation and scheduling overhead without moving any heavy projection or layer compute off ANE.

**Next**: Implement the provider reuse, reusable x-buffer copy, concurrent LM-head shard dispatch, and four-way argmax reduction; then run correctness/residency checks before any powermetrics benchmark.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="024-phi-4-mini-provisional-tok-s-timing-smoke.html">Previous: Journal 024</a> | <a href="026-phi-4-mini-lean-ane-runtime-optimization-outcome.html">Next: Journal 026</a></nav>
