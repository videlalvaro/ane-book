---
layout: default
title: "Journal 039 - Phi-4-mini Four-Layer Fused Shard Golden Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="038-phi-4-mini-four-layer-fused-shard-residency-passed.html">Previous: Journal 038</a> | <a href="040-phi-4-mini-full-4-layer-fused-strategy-intent.html">Next: Journal 040</a></nav>

# 2026-04-27 - Phi-4-mini Four-Layer Fused Shard Golden Passed

**Intent**: Run the next quality gate after the layers [0,4) Phi-4-mini 4-layer fused shard passed strict ANE residency, following the validation-first notes validation-before-scale and whole-operation fusion discipline.

**Setup**: Ran `the range golden validator --layer-start 0 --layer-end 4 --mlmodelc local artifacts --json-out temporary output`; no cleanup/deletion and no energy benchmark.

**Result**: PASS=True. Range golden smoke for [0,4) passed with cos_hidden=0.999597, rmse_hidden=0.020061, and max_abs_hidden=0.269531. The first 4-layer fused Phi-4-mini shard now passes both strict ANE residency and numerical smoke.

**Surprise / hurdle**: The 4-layer fused shard preserved numerical agreement after already proving full ANE residency despite its larger compiled size.

**Lesson**: A single successful 4-layer fused shard is promising, but 4-layer fusion needs representative or full-range validation before runtime migration.

**Next**: Validate representative or all 4-layer ranges before migrating the runtime; do not run energy benchmarking until residency and quality gates hold across that broader set.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="038-phi-4-mini-four-layer-fused-shard-residency-passed.html">Previous: Journal 038</a> | <a href="040-phi-4-mini-full-4-layer-fused-strategy-intent.html">Next: Journal 040</a></nav>
