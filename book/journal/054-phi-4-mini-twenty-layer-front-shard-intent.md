---
layout: default
title: "Journal 054 - Phi-4-mini Twenty-Layer Front Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="053-phi-4-mini-twenty-four-layer-front-shard-intent.html">Previous: Journal 053</a> | <a href="055-phi-4-mini-12-16-20-24-layer-fusion-sweep-outcome.html">Next: Journal 055</a></nav>

# 2026-04-28 - Phi-4-mini Twenty-Layer Front Shard Intent

**Intent**: Before the next Phi-4-mini run, narrow the fusion search from failed [0,24) to [0,20) to locate the strict-residency cliff, applying the validation-first notes Iverson/APL whole-array fusion and Dragon Book call-hoisting only through the established validation gates.

**Setup**: Prior [0,24) compiled at about 2.3G but failed strict residency completely, with all conv and compute ops placed on CPU. Planned non-destructive probe directory: local artifacts; disk is lower at roughly 43 GiB free, so do not delete generated artifacts without explicit confirmation.

**Result**: Intent recorded before execution; no [0,20) artifact, compiled size, residency placement, golden quality, latency, energy, perplexity, or topology result yet.

**Surprise / hurdle**: [0,24) proved that compile success at this size does not imply ANE placement; the next run must identify whether the cliff appears between 20 and 24 layers without using CPU fallback.

**Lesson**: The useful fusion limit is set by strict ANE residency, not merely by CoreML compile success or artifact size.

**Next**: Build/compile [0,20), run strict MLComputePlan residency, then run golden only if residency passes. If gates pass, consider profiling 20+4+6+2 using existing [20,24), [24,30), and [30,32) shards; do not clean up/delete artifacts or modify code for this note.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="053-phi-4-mini-twenty-four-layer-front-shard-intent.html">Previous: Journal 053</a> | <a href="055-phi-4-mini-12-16-20-24-layer-fusion-sweep-outcome.html">Next: Journal 055</a></nav>
