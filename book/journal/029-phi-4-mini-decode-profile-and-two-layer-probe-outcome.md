---
layout: default
title: "Journal 029 - Phi-4-mini Decode Profile and Two-Layer Probe Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="028-phi-4-mini-two-layer-full-shard-probe-intent.html">Previous: Journal 028</a> | <a href="030-phi-4-mini-three-layer-full-shard-probe-intent.html">Next: Journal 030</a></nav>

# 2026-04-27 - Phi-4-mini Decode Profile and Two-Layer Probe Outcome

**Intent**: Measure the decode bottleneck before optimizing, then test a non-destructive 2-layer full-shard probe as a call-count reduction hypothesis, following the validation-first notes measurement-before-optimization and Dragon Book call-hoisting/strength-reduction discipline.

**Setup**: Added decode-only `--profile` breakdown to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); ran existing 32-layer Phi runtime plus LM-head shards. Built separate probe local artifacts and added the range golden validator; no cleanup/deletion.

**Result**: Profile command produced prefill 18.503164s, decode 63 tokens in 7.739233s = 8.140 tok/s, forward 64 in 26.242397s = 2.439 tok/s. Decode-only profile: calls=63, embed_s=0.000065, rope_mask_s=0.000110, layers_s=7.417981, head_copy_s=0.000106, head_predict_reduce_s=0.320883. Per token: embed 0.001 ms, rope/mask 0.002 ms, layers 117.746 ms, head_predict_reduce 5.093 ms; mean layer call 3.680 ms; top5 L1=4.244 ms, L4=4.229 ms, L2=4.137 ms, L15=3.987 ms, L3=3.943 ms. The 2-layer probe package/compiled size was 192M. Residency passed: conv_total=8 conv_ane=8 conv_non_ane=0; compute_total=293 compute_ane=293 compute_non_ane=0. Quality passed: cos=0.999887, rmse=0.008424, max_abs=0.050293.

**Surprise / hurdle**: About 95.9% of decode wall time is the 32 layer CoreML calls; host bookkeeping is negligible and LM-head predict/reduce is about 4.1%.

**Lesson**: Phi-4-mini decode throughput is layer-call dominated, so fusing adjacent layers is the next validated optimization target only if ANE residency and quality remain green.

**Next**: Use the 2-layer probe result to consider bounded fused-layer scale-out; defer powermetrics/energy until after residency and quality remain stable across a larger fused range.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="028-phi-4-mini-two-layer-full-shard-probe-intent.html">Previous: Journal 028</a> | <a href="030-phi-4-mini-three-layer-full-shard-probe-intent.html">Next: Journal 030</a></nav>
