---
layout: default
title: "Journal 027 - Phi-4-mini Decode Profile Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="026-phi-4-mini-lean-ane-runtime-optimization-outcome.html">Previous: Journal 026</a> | <a href="028-phi-4-mini-two-layer-full-shard-probe-intent.html">Next: Journal 028</a></nav>

# 2026-04-27 - Phi-4-mini Decode Profile Intent

**Intent**: Answer the user's concern that about 8 tok/s is still too low by measuring where decode time is lost before further optimization, following measurement-before-optimization discipline.

**Setup**: Planned change: add optional `--profile` timing to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift) around embedding/RoPE-mask host setup, the 32 per-token layer-shard CoreML calls, LM-head input copy, LM-head ANE prediction plus reduction, and per-layer aggregate timings. Scope is lightweight runtime instrumentation only, using existing Phi-4-mini ANE layer and LM-head shards.

**Result**: Intent recorded before implementation; no new latency breakdown, energy, placement, cosine, or perplexity numbers yet.

**Surprise / hurdle**: Aggregate tok/s alone cannot distinguish CoreML layer-call overhead, LM-head dispatch/reduction, host bookkeeping, or a single slow shard.

**Lesson**: Profile the decode pipeline at component granularity before choosing the next optimization target.

**Next**: Implement the optional `--profile` path, run a bounded timing smoke, and keep heavy compute on ANE; do not run powermetrics/energy benchmarking, cleanup, or deletion yet.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="026-phi-4-mini-lean-ane-runtime-optimization-outcome.html">Previous: Journal 026</a> | <a href="028-phi-4-mini-two-layer-full-shard-probe-intent.html">Next: Journal 028</a></nav>
