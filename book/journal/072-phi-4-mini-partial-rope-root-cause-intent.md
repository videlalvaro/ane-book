---
layout: default
title: "Journal 072 - Phi-4-mini Partial-RoPE Root Cause Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="071-phi-private-e5-one-stream-timing-reality-check.html">Previous: Journal 071</a> | <a href="073-phi-lm-head-shard-count-sweep-on-best-topology.html">Next: Journal 073</a></nav>

# 2026-04-29 - Phi-4-mini Partial-RoPE Root Cause Intent

**Intent**: Record the Phi-4-mini generation-quality root cause before the next CoreML probe: the GGUF metadata specifies partial RoPE via `phi3.rope.dimension_count=96`, while the conversion, runtime, and reference stack had been applying RoPE across the full `d_head=128`. The fix follows validation discipline: preserve the real model contract through each layer of the stack before scaling or benchmarking.

**Setup**: Local weights: the local Phi-4-mini GGUF weights. Official HF config reports `partial_rotary_factor=0.75`, matching a 96-dimensional rotary subspace. With the same local GGUF weights, the HF/partial-RoPE path produces valid Erlang for the prompt, while the GGUF-parsed/full-RoPE path produces Python-looking garbage. Code has been patched to carry `rope_dim` through conversion/runtime/reference paths. Planned next run is a user-approved single-layer layer-0 CoreML rebuild probe into a a new temporary directory, followed by compile, golden validation, and strict ANE residency validation.

**Result**: Root cause isolated and intent logged. No new CoreML rebuild, compile, golden, residency, latency, energy, cosine, or perplexity result is recorded in this entry.

**Surprise / hurdle**: The GGUF key used the `phi3.*` namespace and was easy to miss; using `d_head` as the implicit RoPE width made the stack internally consistent enough to build artifacts, but semantically wrong for generation.

**Lesson**: RoPE width is part of the model contract; `d_head` is not a safe default when metadata or HF config defines partial rotary dimensions.

**Next**: Run only the approved layer-0 rebuild probe in a a new temporary directory, then compile and run golden plus strict residency gates before considering broader rebuilds; do not clean up or delete existing artifacts as part of this note.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="071-phi-private-e5-one-stream-timing-reality-check.html">Previous: Journal 071</a> | <a href="073-phi-lm-head-shard-count-sweep-on-best-topology.html">Next: Journal 073</a></nav>
