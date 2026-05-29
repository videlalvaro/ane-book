---
layout: default
title: "Journal 022 - Phi-4-mini Runtime Scaffolding Smoke Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="021-phi-4-mini-lm-head-shards-passed.html">Previous: Journal 021</a> | <a href="023-phi-4-mini-tok-s-timing-smoke-intent.html">Next: Journal 023</a></nav>

# 2026-04-27 - Phi-4-mini Runtime Scaffolding Smoke Passed

**Intent**: Continue from completed layer shards and LM-head shards into prompt-ID runtime scaffolding while preserving the ANE-only boundary and validation-before-performance discipline.

**Setup**: Added [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py) to export the permitted host embedding lookup bin plus runtime manifest. Added [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift), a prompt-ID smoke runtime chaining 32 stateful layer shards and 4 ANE LM-head shards, with host work limited to embedding lookup, RoPE/mask bookkeeping, and argmax. Compiled with a local Swift/CoreML compile command.

**Result**: Wrote local embedding weights (1.1G), the runtime metadata (4.1K), and the runtime binary (148K). The full-chain smoke command loaded all 32 layers plus 4 LM-head shards and generated next token ID 6360.

**Surprise / hurdle**: Runtime integration could be tested without tokenizer integration by using prompt IDs directly, keeping host work inside the permitted non-compute exceptions.

**Lesson**: Once all layer and LM-head shards exist, a minimal prompt-ID runtime can prove artifact chaining before spending time on tokenizer, benchmarking, or full-logit golden validation.

**Next**: No perf/energy benchmarking, tokenizer integration, HF full-logit golden validation, cleanup, or deletion was performed; those remain separate gated steps.

**Refs**: [converters/phi4_mini_export_runtime.py](https://github.com/videlalvaro/ane-book/blob/main/converters/phi4_mini_export_runtime.py); [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="021-phi-4-mini-lm-head-shards-passed.html">Previous: Journal 021</a> | <a href="023-phi-4-mini-tok-s-timing-smoke-intent.html">Next: Journal 023</a></nav>
