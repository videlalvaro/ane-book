---
layout: default
title: "Journal 038 - Phi-4-mini Four-Layer Fused Shard Residency Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="037-phi-4-mini-four-layer-fused-shard-intent.html">Previous: Journal 037</a> | <a href="039-phi-4-mini-four-layer-fused-shard-golden-passed.html">Next: Journal 039</a></nav>

# 2026-04-27 - Phi-4-mini Four-Layer Fused Shard Residency Passed

**Intent**: Test whether a larger Phi-4-mini fused shard for layers [0,4) can remain ANE-resident beyond the prior 3-layer/288M probe, following the validation-first notes validation-before-scale and whole-operation fusion discipline.

**Setup**: Built non-destructively under local artifacts; converted and compiled one INT8 stateful CoreML shard for layers [0,4); ran strict MLComputePlan residency. Residency JSON: temporary output.

**Result**: PASS. Conversion and compile succeeded. Artifact sizes: `.mlpackage` 384M and `.mlmodelc` 385M. Strict residency passed: conv_total=16 conv_ane=16 conv_non_ane=0; compute_total=583 compute_ane=583 compute_non_ane=0; PASS=True.

**Surprise / hurdle**: A single 4-layer shard stayed ANE-resident despite exceeding both the prior 3-layer 288M probe and the conservative ~250M compiled-shard guidance.

**Lesson**: Phi-4-mini fused-shard size limits are empirical and range-specific; exceeding conservative size guidance does not imply ANE fallback when strict residency still passes.

**Next**: Run golden validation for [0,4), then validate representative/all 4-layer ranges before any runtime migration. No cleanup/deletion and no energy benchmark were performed.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="037-phi-4-mini-four-layer-fused-shard-intent.html">Previous: Journal 037</a> | <a href="039-phi-4-mini-four-layer-fused-shard-golden-passed.html">Next: Journal 039</a></nav>
