---
layout: default
title: "Journal 084 - Phi Public Algorithmic Perf Direction Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="083-phi-dead-artifact-cleanup-outcome.html">Previous: Journal 083</a> | <a href="085-phi-n-gram-proposal-probe-added.html">Next: Journal 085</a></nav>

# 2026-04-29 - Phi Public Algorithmic Perf Direction Intent

**Intent**: Pivot away from private ANE APIs for small wins; pursue public, ANE-only algorithmic performance via n-gram acceptance, speculative decoding, prompt-lookup decoding, and related public-runtime approaches, following measurement-before-optimization discipline.

**Setup**: Planning note only. Current Phi Swift runtime uses mutable `MLState` with single-token layer shards and the preserved 20+4+6+2 baseline plus batch-4 LM-head artifacts. No command run, no conversion, no benchmark, no cleanup/deletion.

**Result**: Direction recorded; no placement, latency, energy, cosine, perplexity, or acceptance-rate numbers yet.

**Surprise / hurdle**: Exact speculative batch verification cannot be dropped into the current runtime blindly because mutable CoreML state would need rollback/copy semantics, or separate batch-capable layer artifacts, before multiple candidate tokens can be verified without corrupting the greedy KV path.

**Lesson**: Public ANE-only speedups should first measure acceptance opportunity and state-management cost before changing the baseline greedy decode path.

**Next**: Implement only an opt-in proposal/accounting probe first: estimate n-gram/speculative/prompt-lookup acceptance potential and accounting overhead while leaving baseline greedy generation unchanged. The cleanup journal changes are currently uncommitted and should be committed with this work or separately.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="083-phi-dead-artifact-cleanup-outcome.html">Previous: Journal 083</a> | <a href="085-phi-n-gram-proposal-probe-added.html">Next: Journal 085</a></nav>
