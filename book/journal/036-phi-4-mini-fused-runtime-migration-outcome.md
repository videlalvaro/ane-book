---
layout: default
title: "Journal 036 - Phi-4-mini Fused Runtime Migration Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="035-phi-4-mini-fused-runtime-migration-intent.html">Previous: Journal 035</a> | <a href="037-phi-4-mini-four-layer-fused-shard-intent.html">Next: Journal 037</a></nav>

# 2026-04-27 - Phi-4-mini Fused Runtime Migration Outcome

**Intent**: Complete the runtime migration from 32 one-layer Phi-4-mini shards to the validated 11 fused layer shards, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py) now supports `--layer-artifact-dir`, `--layer-group-size`, and `--manifest-name`; [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift) now validates contiguous layer coverage instead of `layer_shards == n_layers` and reports fused range timings. Generated local artifacts with 11 layer entries covering 0..32 and paths to `../phi4_mini_ane_3layer_probe/*.mlmodelc`; compiled the runtime and profiled with the fused manifest.

**Result**: 64-token run: layer_shards=11, decode_tokens=63, decode_s=4.850130, decode_tok_s=12.989, layers_ms/token=71.837, head_ms/token=5.143. Sustained 128-token exact runs observed best decode_tok_s=14.332 with layers_ms/token=64.690 and head_ms/token=5.076; later exact run decode_tok_s=12.757 with layers_ms/token=71.145 and head_ms/token=7.238. Versus the ~8.0 tok/s 32-shard baseline, fused runtime is roughly 1.6–1.8x faster steady decode.

**Surprise / hurdle**: First-token/prefill rose to ~88–91s because large fused-shard loading/warmup is included; steady decode is the relevant metric for this comparison.

**Lesson**: Matching the runtime topology to validated fused ANE shards materially reduces per-token layer-call overhead while keeping heavy compute ANE-only.

**Next**: No cleanup/deletion and no energy benchmark were performed; next gated step is energy measurement or further fused-runtime profiling with the same ANE-only boundary.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md); [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py); [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="035-phi-4-mini-fused-runtime-migration-intent.html">Previous: Journal 035</a> | <a href="037-phi-4-mini-four-layer-fused-shard-intent.html">Next: Journal 037</a></nav>
