---
layout: default
title: "Journal 033 - LLM.int8() ANE Conv2D Adaptation Review Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="032-phi-4-mini-full-3-layer-shard-strategy-validation-intent.html">Previous: Journal 032</a> | <a href="034-phi-4-mini-full-3-layer-shard-strategy-validated.html">Next: Journal 034</a></nav>

# 2026-04-27 - LLM.int8() ANE Conv2D Adaptation Review Intent

**Intent**: Review Dettmers et al. LLM.int8() (arXiv:2208.07339) and assess whether vector-wise quantization plus mixed-precision outlier decomposition can be adapted to the repository's Conv2D(1x1)-based ANE conversion path. The analysis follows validation-before-scale discipline: first reason about operator shape, quantization semantics, and ANE residency risk before proposing any implementation.

**Setup**: Planning task only. Scope is paper/codepath analysis against the existing CoreML Conv2D(1x1) ANE shard strategy, current INT8 per-tensor production baseline, and ANE residency/golden quality gates. No conversion, compilation, benchmarking, cleanup, deletion, or other destructive operation should be run for this note.

**Result**: Intent recorded before analysis; no artifacts produced, no commands run, and no placement, latency, energy, cosine, or perplexity numbers yet.

**Surprise / hurdle**: LLM.int8() relies on vector-wise quantization and explicit high-precision outlier handling, so the open question is whether its decomposition can be represented as ANE-resident Conv2D(1x1) shards without introducing CPU/GPU fallback or host-side compute.

**Lesson**: Mixed-precision quantization ideas are useful for this project only if both the main quantized path and the outlier path remain CoreML/ANE-resident and pass golden quality before scale-out.

**Next**: Read arXiv:2208.07339, map its vector-wise and outlier decomposition steps onto the repository's Conv2D(1x1) conversion constraints, identify a smallest representative ANE residency probe if promising, and keep the work analysis-only until a separate gated implementation intent is approved.

**Refs**: [arXiv:2208.07339](https://arxiv.org/abs/2208.07339); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="032-phi-4-mini-full-3-layer-shard-strategy-validation-intent.html">Previous: Journal 032</a> | <a href="034-phi-4-mini-full-3-layer-shard-strategy-validated.html">Next: Journal 034</a></nav>
