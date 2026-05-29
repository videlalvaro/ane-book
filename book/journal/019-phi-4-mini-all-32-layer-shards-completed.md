---
layout: default
title: "Journal 019 - Phi-4-mini All 32 Layer Shards Completed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="018-phi-4-mini-remaining-layers-16-31-run-batches-intent.html">Previous: Journal 018</a> | <a href="020-phi-4-mini-lm-head-shard-builder-intent.html">Next: Journal 020</a></nav>

# 2026-04-27 - Phi-4-mini All 32 Layer Shards Completed

**Intent**: Complete all Phi-4-mini layer shards because the user explicitly requested all layers built to completion, while preserving validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: Ran the remaining-layer batch stage of the Phi orchestration script, with a continuation/follow-up for tail layers when needed. Artifacts targeted the local artifact directory; per-layer gates were convert, compile, strict MLComputePlan residency, and numerical smoke.

**Result**: PASS. Final verification of the generated local artifacts: 32/32 `.mlpackage`, 32/32 `.mlmodelc`, and 32/32 `_meta.json` exist; all 32 residency reports and all 32 golden reports exist; missing_count=0; failed_residency_layers=[]; failed_golden_layers=[]; total artifact directory size 6.0G. Strict residency for every layer: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Numerical smoke across layers: cos_hidden min=0.9997643231 max=0.9999581553 mean=0.9998512843; rmse_hidden min=0.0047372482 max=0.04397358 mean=0.0211910720; max_abs_hidden min=0.0263671875 max=0.359375 mean=0.1135444641.

**Surprise / hurdle**: Completion required an explicit remaining-layer command plus tail follow-up rather than an unchecked full conversion; the artifact audit found no missing or failed layer gates.

**Lesson**: Explicit user-requested completion can scale a guarded ANE conversion to all Phi-4-mini layers without observed residency fallback when every layer remains individually gated.

**Next**: No perf/energy benchmarking, no full-model runtime/golden logits, no LM head conversion, and no cleanup/deletion were performed; those remain separate gated steps.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="018-phi-4-mini-remaining-layers-16-31-run-batches-intent.html">Previous: Journal 018</a> | <a href="020-phi-4-mini-lm-head-shard-builder-intent.html">Next: Journal 020</a></nav>
