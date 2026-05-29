---
layout: default
title: "Journal 028 - Phi-4-mini Two-Layer Full-Shard Probe Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="027-phi-4-mini-decode-profile-intent.html">Previous: Journal 027</a> | <a href="029-phi-4-mini-decode-profile-and-two-layer-probe-outcome.html">Next: Journal 029</a></nav>

# 2026-04-27 - Phi-4-mini Two-Layer Full-Shard Probe Intent

**Intent**: Test a 2-layer full-shard probe covering layers 0–2 to reduce decode layer CoreML calls from 32 to 16 if ANE residency and quality pass, applying Dragon Book strength reduction/call-hoisting and Iverson whole-operation fusion.

**Setup**: Decode-only profile showed Phi-4-mini spends about 117.746 ms/token in 32 layer CoreML calls, mean about 3.680 ms/layer call; LM head is about 5.093 ms/token and host bookkeeping is negligible. Planned probe uses INT8, a separate output directory, and no cleanup or deletion.

**Result**: Intent recorded before implementation; no 2-layer shard artifacts, residency numbers, quality cosine/perplexity, latency, or energy results yet.

**Surprise / hurdle**: The current bottleneck is per-layer CoreML call count rather than LM-head or host bookkeeping, so the next optimization must fuse layers without losing ANE placement or numerical quality.

**Lesson**: When launch/call overhead dominates decode, whole-operation layer fusion is the safe next hypothesis only if residency and golden quality gates remain mandatory.

**Next**: Build the layers 0–2 INT8 full-shard probe in a separate output directory, run ANE residency and quality gates, and defer energy/powermetrics until those pass.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="027-phi-4-mini-decode-profile-intent.html">Previous: Journal 027</a> | <a href="029-phi-4-mini-decode-profile-and-two-layer-probe-outcome.html">Next: Journal 029</a></nav>
