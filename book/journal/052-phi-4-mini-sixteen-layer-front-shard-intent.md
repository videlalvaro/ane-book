---
layout: default
title: "Journal 052 - Phi-4-mini Sixteen-Layer Front Shard Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="051-phi-4-mini-twelve-layer-front-shard-intent.html">Previous: Journal 051</a> | <a href="053-phi-4-mini-twenty-four-layer-front-shard-intent.html">Next: Journal 053</a></nav>

# 2026-04-28 - Phi-4-mini Sixteen-Layer Front Shard Intent

**Intent**: Before the next Phi-4-mini run, test whether a larger front fused shard [0,16) can improve on the valid 12+12+6+2 topology, applying the validation-first notes Iverson/APL whole-array fusion and Dragon Book call-hoisting discipline while keeping validation ahead of performance claims.

**Setup**: Current comparison point: 12+12+6+2 is valid and produced profiles of 16.598 tok/s and then 17.159 tok/s, making it a possible new best that still needs controlled comparison. Planned non-destructive probe directory: local artifacts; proposed topology if gates pass is 16+8+6+2.

**Result**: Intent recorded before execution; no [0,16) artifact, residency placement, golden quality, latency comparison, energy, or perplexity result yet.

**Surprise / hurdle**: The late [24,32) tail remains forbidden as a single 8-layer fused shard because prior golden validation produced NaN despite ANE residency, so any usable larger topology must keep the tail split.

**Lesson**: Treat 12+12+6+2 as promising but provisional; larger fusion is useful only if the exact [0,16) shard passes build, compile, strict residency, and golden without NaN or non-ANE fallback.

**Next**: Gate order is unchanged: build, compile, MLComputePlan residency, then golden. Use the 16+8+6+2 topology only if residency and golden pass; do not use any NaN or non-ANE result, and do not clean up/delete artifacts or run energy benchmarking for this intent note.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="051-phi-4-mini-twelve-layer-front-shard-intent.html">Previous: Journal 051</a> | <a href="053-phi-4-mini-twenty-four-layer-front-shard-intent.html">Next: Journal 053</a></nav>
