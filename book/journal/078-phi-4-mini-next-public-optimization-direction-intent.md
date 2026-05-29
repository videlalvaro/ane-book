---
layout: default
title: "Journal 078 - Phi-4-mini Next Public Optimization Direction Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="077-phi-20-5-5-2-tail-probe.html">Previous: Journal 077</a> | <a href="079-phi-lm-head-top-k-shard-residency-failed.html">Next: Journal 079</a></nav>

# 2026-04-28 - Phi-4-mini Next Public Optimization Direction Intent

**Intent**: After establishing the public Phi-4-mini baseline topology `20+4+6+2` and rejecting `20+5+5+2` as slower, start the next book-shaped ANE optimization direction. The two likely probes are Iverson/APL-style fatter token shapes (`T>1` layer-shard inputs, treating more token work as one array operation) and Stepanov-style hierarchical LM-head reduction (using associative reduction structure to reduce projection/result handling depth).

**Setup**: Planning note only. Existing public CoreML Phi-4-mini topology comparison is the starting point; proposed probes must use CoreML `.mlpackage` artifacts targeting ANE for compute-heavy work. Host work remains limited to permitted bookkeeping/sampling/string/file tasks; no CPU/GPU matmul, projection, norm, attention, FFN, or LM-head compute shortcut is acceptable.

**Result**: Intent recorded before implementation. No new artifacts, placement numbers, latency, energy, cosine, perplexity, or topology result yet.

**Surprise / hurdle**: The public topology search is in a diminishing-returns region where nearby shard shapes can become slower, so the next optimization should change the problem shape rather than only nudge layer group sizes.

**Lesson**: When fused-layer topology gains plateau, move to array-shape and reduction-structure probes, but keep every heavy compute path ANE-resident and gated before scale-out.

**Next**: Design the smallest representative gate for either `T>1` layer-shard inputs or hierarchical LM-head reduction; run strict ANE residency and golden quality before any broader build, runtime migration, performance claim, energy benchmark, cleanup, or deletion.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="077-phi-20-5-5-2-tail-probe.html">Previous: Journal 077</a> | <a href="079-phi-lm-head-top-k-shard-residency-failed.html">Next: Journal 079</a></nav>
