---
layout: default
title: "Journal 094 - Phi-4-mini Real-Weight T=4 Verifier Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="093-phi-t-4-verifier-op-pattern-probe-passed.html">Previous: Journal 093</a> | <a href="095-phi-4-mini-real-weight-t-4-verifier-layer-passed.html">Next: Journal 095</a></nav>

# 2026-04-29 - Phi-4-mini Real-Weight T=4 Verifier Intent

**Intent**: After checkpoint `b366672` was committed and tagged as `phi4-mini-ane-v0-spec-2026-04-29`, start the next public Phi-4-mini ANE experiment: implement and test a real-weight one-layer `T=4` verifier shard. The hypothesis follows Experiment 26, combining Dragon Book data-flow invariants, Knuth sequential verification, and the speculative decoding verifier framing from Leviathan et al. (2023): one target pass should verify several draft tokens only if the block graph exactly matches four sequential single-token target calls.

**Setup**: Planned path is public CoreML/ANE only, with no unsupported runtime path and no CPU/GPU compute fallback. Build only one real Phi-4-mini layer first to conserve disk/RAM. Target verifier shape keeps compute-heavy work inside a CoreML `.mlpackage`: `x [1,d,T,1]`, per-token RoPE rows, causal `attn_mask [1,1,T,max_seq]`, `kv_write_mask [1,1,max_seq,T]`, stateful multi-row KV write, and hidden output `[1,d,T,1]` for `T=4`. Acceptance gates are: compile one real layer, compare its block output/state behavior against four sequential single-token Phi calls, then run strict MLComputePlan residency.

**Result**: Intent recorded before implementation. No real-weight verifier artifacts, parity numbers, residency counts, latency, energy, cosine, perplexity, or scale-out results yet.

**Surprise / hurdle**: The Phi-sized synthetic `T=4` KV scatter probe passed ANE residency, but that only proves the op family at representative shape; real weights, real RoPE/KV semantics, and exact sequential parity remain unproven and must be checked before spending disk/RAM on more layers.

**Lesson**: Synthetic ANE residency is permission to try one real-weight shard, not permission to scale; the real verifier is accepted only when four-token parity and strict ANE residency both pass.

**Next**: Implement the one-layer real-weight `T=4` verifier shard, compare against four sequential single-token Phi calls, run strict MLComputePlan residency, and stop there unless both parity and ANE residency pass. Do not scale to all layers, do not benchmark performance/energy, and do not introduce unsupported runtime path or CPU/GPU compute fallback for this experiment.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="093-phi-t-4-verifier-op-pattern-probe-passed.html">Previous: Journal 093</a> | <a href="095-phi-4-mini-real-weight-t-4-verifier-layer-passed.html">Next: Journal 095</a></nav>
