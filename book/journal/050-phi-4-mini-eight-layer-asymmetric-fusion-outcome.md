---
layout: default
title: "Journal 050 - Phi-4-mini Eight-Layer Asymmetric Fusion Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="049-phi-4-mini-eight-layer-fused-shard-intent.html">Previous: Journal 049</a> | <a href="051-phi-4-mini-twelve-layer-front-shard-intent.html">Next: Journal 051</a></nav>

# 2026-04-28 - Phi-4-mini Eight-Layer Asymmetric Fusion Outcome

**Intent**: Validate whether 8-layer Phi-4-mini fused INT8 stateful CoreML shards can reduce layer-chain overhead beyond the 6-layer baseline, applying the validation-first notes Iverson/APL whole-operation fusion and Dragon Book call-hoisting/strength-reduction discipline while preserving strict ANE residency and golden quality gates.

**Setup**: Built full 8-layer compiled artifacts under local artifacts; each compiled artifact is about 769 MB. Validated ranges [0,8), [8,16), [16,24), and [24,32), then generated the successful asymmetric runtime manifest local artifacts using [0,8), [8,16), [16,24), [24,30), and [30,32). The exporter now supports repeated `--layer-spec start:end:path` arguments for asymmetric manifests.

**Result**: Strict residency/golden results: [0,8) passed golden with cos=0.9993929686, rmse=0.03399, max_abs=0.18848; [8,16) passed golden with cos=0.9983014692, rmse=0.06716, max_abs=0.265625; [16,24) passed strict residency with conv_total=32 conv_ane=32 compute_non_ane=0 and golden cos=0.9989950699, rmse=0.130797, max_abs=0.6328125. The [24,32) shard was ANE-resident but golden failed with NaN, so the full 8/8/8/8 topology is not usable. Asymmetric runtime profiling with the 8/8/8/6/2 manifest: run1 decode_tok_s=16.622, layers_ms/token=55.057, head_predict_reduce_ms=5.095; run2 decode_tok_s=16.653, layers_ms/token=54.959, head_predict_reduce_ms=5.082. Prior 6-layer baseline was about 15.4-16.1 tok/s with layers around 59.5-60 ms/token.

**Surprise / hurdle**: The tail [24,32) range remained ANE-resident but produced NaN in golden validation, proving that residency alone is insufficient for fused-topology acceptance and forcing an asymmetric tail split.

**Lesson**: Larger layer fusion can improve Phi-4-mini decode throughput, but topology selection must be driven by both residency and golden validation; the late tail cannot be fused as a single 8-layer shard.

**Next**: Probe larger front or middle ranges only behind strict residency and golden gates; keep the tail split because [24,32) cannot be used as an 8-layer fused shard due to NaN.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="049-phi-4-mini-eight-layer-fused-shard-intent.html">Previous: Journal 049</a> | <a href="051-phi-4-mini-twelve-layer-front-shard-intent.html">Next: Journal 051</a></nav>
