---
layout: default
title: "Journal 100 - Phi-4-mini Partial-RoPE Patch and Probe Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="099-phi-full-stack-gguf-reference-gate-blocks-q8-chat.html">Previous: Journal 099</a> | <a href="101-phi-4-mini-rope96-fast-fused-rebuild-intent.html">Next: Journal 101</a></nav>

# 2026-04-29 - Phi-4-mini Partial-RoPE Patch and Probe Passed

**Intent**: Record the confirmed root cause and code patch for bad Phi-4-mini chat output, following validation discipline: preserve the official model contract before optimizing or rebuilding production artifacts. The specific contract is partial RoPE from the official `microsoft/Phi-4-mini-instruct` config and matching GGUF metadata.

**Setup**: Local weights: the local Phi-4-mini GGUF weights. Official HF config has `partial_rotary_factor=0.75`, and GGUF metadata has `phi3.rope.dimension_count=96`; the previous conversion/runtime/reference stack incorrectly rotated the full `d_head=128`. Code was patched to carry `rope_dim` and use `rope_dim//2` cos/sin in [converters/gguf_to_ane.py](https://github.com/videlalvaro/ane-book/blob/main/converters/gguf_to_ane.py), [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift), [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py), helper script, the layer golden validator, and helper script. User-approved probe directory: temporary output; scope was single-layer only.

**Result**: Root cause confirmed. With official partial-RoPE config and the same local GGUF, prompt `write hello world in Erlang` produces a valid Erlang module; with the old full-RoPE config it produces Python-looking garbage. The patched reference now generates an Erlang code fence containing `io:format("Hello, World!~n", []).`. The single-layer layer-0 q8 mlpackage was built and compiled; golden passed with `cos=0.999958`, `rmse=0.004737`, `max_abs=0.026367`; multi-token positions 0..3 passed; strict residency passed with `conv_ane=4/4` and `compute_ane=152/152`.

**Surprise / hurdle**: The older stack was internally self-consistent enough for local shard gates to pass, yet wrong at the semantic model-contract level because it ignored the partial rotary subspace. The GGUF key lives under the `phi3.*` namespace, which made the missing `rope_dim=96` easy to overlook.

**Lesson**: Full-stack chat quality can fail from a metadata contract mismatch even when layer-local ANE residency and golden smoke tests are green; RoPE dimension must be propagated explicitly through conversion, reference, runtime, and validation code.

**Next**: Existing production Phi q8 artifacts are still old full-RoPE graphs and must be rebuilt before chat can work. Do not treat the patched code alone as fixing deployed chat artifacts; rebuild only through the normal ANE residency and full-stack quality gates.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="099-phi-full-stack-gguf-reference-gate-blocks-q8-chat.html">Previous: Journal 099</a> | <a href="101-phi-4-mini-rope96-fast-fused-rebuild-intent.html">Next: Journal 101</a></nav>
