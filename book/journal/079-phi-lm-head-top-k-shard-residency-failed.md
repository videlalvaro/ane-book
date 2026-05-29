---
layout: default
title: "Journal 079 - Phi LM-Head Top-K Shard Residency Failed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="078-phi-4-mini-next-public-optimization-direction-intent.html">Previous: Journal 078</a> | <a href="080-phi-batch-4-lm-head-shape-probe-passed.html">Next: Journal 080</a></nav>

# 2026-04-28 - Phi LM-Head Top-K Shard Residency Failed

**Intent**: Verify the existing Phi LM-head top-k shard before any scale-out, following the ANE-only mandate and validation-before-scale discipline.

**Setup**: Checked compiled artifact `Phi4MiniLMHead_top1_s0_q8.mlmodelc` with strict residency validation.

**Result**: FAIL. Residency reported conv_total=1 conv_ane=1, but compute_total=11 compute_ane=9 compute_non_ane=2, PASS=False. The non-ANE ops were `ios18.topk` and `ios18.cast` on CPU.

**Surprise / hurdle**: The convolution stayed ANE-resident, but the top-k reduction pattern introduced CPU fallback through CoreML lowering.

**Lesson**: Do not scale the top-k LM-head path under the ANE-only mandate when `topk`/`cast` fall back to CPU.

**Next**: Pivot to a batched LM-head projection shape probe instead of scaling this top-k artifact.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="078-phi-4-mini-next-public-optimization-direction-intent.html">Previous: Journal 078</a> | <a href="080-phi-batch-4-lm-head-shape-probe-passed.html">Next: Journal 080</a></nav>
