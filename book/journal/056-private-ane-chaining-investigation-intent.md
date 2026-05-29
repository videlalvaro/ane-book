---
layout: default
title: "Journal 056 - Private ANE Chaining Investigation Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="055-phi-4-mini-12-16-20-24-layer-fusion-sweep-outcome.html">Previous: Journal 055</a> | <a href="057-private-ane-api-bridge-outcome.html">Next: Journal 057</a></nav>

# 2026-04-28 - Private ANE Chaining Investigation Intent

**Intent**: After validating the public CoreML fused Phi-4-mini topology at about 17.1 decode tok/s, start investigating private/Internal ANE API chaining to avoid CoreML per-shard hidden-state roundtrips without relying solely on larger layer fusion. The hypothesis follows call-hoisting/strength-reduction discipline: remove boundary crossings while preserving ANE-only compute.

**Setup**: Planning note before the run. Knowledge source: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md). First target: a small proof-of-concept that chains two already-validated ANE layer shards while keeping intermediates off the Swift/CoreML boundary. Existing public baseline remains the strict-resident, golden-passed Phi-4-mini 16+8+6+2 CoreML fused topology.

**Result**: Intent recorded before execution; no private API probe, artifact, placement result, latency, energy, cosine, perplexity, cleanup, deletion, or code change has been run for this entry.

**Surprise / hurdle**: Public CoreML fusion improves tok/s but still exposes per-shard boundary costs; private/Internal chaining may reduce those costs, but must not replace public residency and golden quality gates.

**Lesson**: The next runtime hypothesis is ANE-side chaining of validated shards, not further CPU/GPU host optimization or unchecked fusion.

**Next**: Use `ane-internals` as research context, build only a minimal two-shard chaining proof-of-concept, and require strict ANE residency plus golden comparison before any broader scale-out or performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="055-phi-4-mini-12-16-20-24-layer-fusion-sweep-outcome.html">Previous: Journal 055</a> | <a href="057-private-ane-api-bridge-outcome.html">Next: Journal 057</a></nav>
