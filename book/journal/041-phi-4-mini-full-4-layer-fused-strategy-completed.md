---
layout: default
title: "Journal 041 - Phi-4-mini Full 4-Layer Fused Strategy Completed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="040-phi-4-mini-full-4-layer-fused-strategy-intent.html">Previous: Journal 040</a> | <a href="042-phi-4-mini-isolated-warm-cache-outcome.html">Next: Journal 042</a></nav>

# 2026-04-27 - Phi-4-mini Full 4-Layer Fused Strategy Completed

**Intent**: Complete validation and runtime profiling of the full Phi-4-mini 4-layer fused-shard strategy, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Built and compiled all eight 4-layer ranges [0,4), [4,8), [8,12), [12,16), [16,20), [20,24), [24,28), and [28,32) under local artifacts; each mlpackage/mlmodelc was about 384-385 MB. Generated local artifacts and profiled the Swift runtime with the 4-layer manifest.

**Result**: PASS. Strict MLComputePlan residency passed for all 8 ranges: conv_total=16 conv_ane=16 conv_non_ane=0; compute_total=583 compute_ane=583 compute_non_ane=0. Range golden passed for all 8: cos min/mean/max=0.999342/0.999508/0.999688, rmse max=0.112766, max_abs max=0.500000. Best 4-layer repeat profile: decode_tokens=127, decode_s=8.265830, decode_tok_s=15.364; ProfileDecodePerToken layers_ms=59.988, head_predict_reduce_ms=5.091; layer_shards=8 mean_layer_shard_call_ms=7.499. Same-machine 3-layer comparison: decode_tok_s=14.358, layers_ms=64.565, head_predict_reduce_ms=5.079.

**Surprise / hurdle**: Four-layer fusion validated despite 384-385 MB shard artifacts, but shard-call granularity is now in diminishing returns; LM-head prediction/reduction remains about 5 ms/token.

**Lesson**: Four-layer Phi-4-mini fusion is validated and modestly faster than the 3-layer runtime, but the remaining bottlenecks are total layer compute/call time and LM-head prediction/reduction rather than host bookkeeping.

**Next**: No energy benchmark was run; future work should target layer compute/call total and LM-head prediction/reduction while preserving ANE residency and golden quality gates.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="040-phi-4-mini-full-4-layer-fused-strategy-intent.html">Previous: Journal 040</a> | <a href="042-phi-4-mini-isolated-warm-cache-outcome.html">Next: Journal 042</a></nav>
