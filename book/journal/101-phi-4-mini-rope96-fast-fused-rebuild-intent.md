---
layout: default
title: "Journal 101 - Phi-4-mini Rope96 Fast Fused Rebuild Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="100-phi-4-mini-partial-rope-patch-and-probe-passed.html">Previous: Journal 100</a> | <a href="102-phi-4-mini-rope96-fast-fused-rebuild-outcome.html">Next: Journal 102</a></nav>

# 2026-04-30 - Phi-4-mini Rope96 Fast Fused Rebuild Intent

**Intent**: Rebuild the Phi-4-mini fast fused runtime topology with the already-fixed partial RoPE contract (`rope_dim=96`) so the old fastest public path can be tested with correct model semantics. This follows validation-before-performance discipline plus Dragon Book call-hoisting/strength-reduction and Iverson whole-operation fusion: keep the fused topology for lower CoreML call count, but require the corrected RoPE metadata contract before trusting throughput or chat behavior.

**Setup**: Planned source artifact: the local Phi-4-mini GGUF weights. Planned output: local artifacts with INT8 per-tensor stateful CoreML shards for topology [0,20)+[20,24)+[24,30)+[30,32). Per fused shard gates: compile, strict ANE placement via [validators/phi4_mini_residency_check.py](https://github.com/videlalvaro/ane-book), and range golden via the range golden validator. After shard gates pass, planned runtime export is `phi4mini_runtime_meta_rope96_fast_20_4_6_2.json`, followed by compile/use of the existing rope96 Swift runtime and an Erlang hello-world smoke test.

**Result**: Intent recorded before the non-trivial rebuild. No new artifacts, compile status, residency numbers, cosine/RMSE, latency, energy, or full-stack smoke results yet.

**Surprise / hurdle**: The single-layer rope96 path is already the correctness baseline and must not be disturbed; the risk is rebuilding the production-speed fused path without regressing the partial-RoPE fix or relying on old full-RoPE artifacts.

**Lesson**: A fast fused topology is useful only after the model metadata contract, especially partial RoPE, is rebuilt into every ANE shard and re-gated end to end.

**Next**: Rebuild only the [0,20)+[20,24)+[24,30)+[30,32) fused shards from the GGUF, run compile/residency/range-golden gates per shard, export `phi4mini_runtime_meta_rope96_fast_20_4_6_2.json`, compile/run the rope96 Swift runtime, and record the Erlang hello-world smoke outcome in a follow-up entry.

**Refs**: [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book); [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="100-phi-4-mini-partial-rope-patch-and-probe-passed.html">Previous: Journal 100</a> | <a href="102-phi-4-mini-rope96-fast-fused-rebuild-outcome.html">Next: Journal 102</a></nav>
