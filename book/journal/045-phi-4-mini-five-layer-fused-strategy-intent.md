---
layout: default
title: "Journal 045 - Phi-4-mini Five-Layer Fused Strategy Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="044-phi-4-mini-resident-serve-mode-landed.html">Previous: Journal 044</a> | <a href="046-phi-4-mini-six-layer-fused-strategy-intent.html">Next: Journal 046</a></nav>

# 2026-04-28 - Phi-4-mini Five-Layer Fused Strategy Intent

**Intent**: After the user asked whether tok/s can be pushed higher and the 4-layer fused runtime reached about 15 tok/s, scale the successful single 5-layer Phi-4-mini fused shard probe non-destructively to the full 5-layer topology, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Existing probe: single fused INT8 shard [0,5) under local artifacts, built and compiled at 481M. Planned ranges: [0,5), [5,10), [10,15), [15,20), [20,25), [25,30), and tail [30,32). Run strict MLComputePlan residency and range golden for every range before generating any 5-layer runtime manifest/profile.

**Result**: Intent recorded after representative probe passed. Probe residency passed: conv_total=20 conv_ane=20 conv_non_ane=0; compute_total=728 compute_ane=728 compute_non_ane=0. Probe range golden passed: cos_hidden=0.999532, rmse_hidden=0.027086, max_abs_hidden=0.281250.

**Surprise / hurdle**: The 481M 5-layer compiled shard exceeded earlier conservative shard-size guidance yet still passed strict residency and range golden for [0,5), so all remaining ranges must re-prove compile success, ANE residency, and numerical quality before runtime migration.

**Lesson**: Larger fused shards can improve tok/s only after every planned range independently passes ANE residency and golden gates; a single representative pass is an invitation to validate, not a scale-out proof.

**Next**: Build/compile the remaining 5-layer ranges under local artifacts, then run strict residency and range golden for all ranges; do not delete/clean up artifacts and do not run energy benchmarking.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="044-phi-4-mini-resident-serve-mode-landed.html">Previous: Journal 044</a> | <a href="046-phi-4-mini-six-layer-fused-strategy-intent.html">Next: Journal 046</a></nav>
