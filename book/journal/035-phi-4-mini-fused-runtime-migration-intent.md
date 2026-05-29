---
layout: default
title: "Journal 035 - Phi-4-mini Fused Runtime Migration Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="034-phi-4-mini-full-3-layer-shard-strategy-validated.html">Previous: Journal 034</a> | <a href="036-phi-4-mini-fused-runtime-migration-outcome.html">Next: Journal 036</a></nav>

# 2026-04-27 - Phi-4-mini Fused Runtime Migration Intent

**Intent**: After the full Phi-4-mini 3-layer fused-shard strategy passed residency and golden across all ranges, migrate the runtime from 32 one-layer shards to the validated 11 fused layer shards, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Planned work: add/export a fused runtime manifest, update the Swift runtime to accept contiguous layer ranges where shard count does not equal `n_layers`, compile the runtime, and re-profile tok/s. Heavy compute remains ANE-only; permitted host work stays limited to embedding lookup, RoPE/mask bookkeeping, sampling, metadata, and cache position bookkeeping. No cleanup/deletion and no energy/powermetrics unless explicitly requested.

**Result**: Intent recorded before implementation; no new compiled runtime, tok/s, energy, placement, cosine, or perplexity numbers yet.

**Surprise / hurdle**: The old runtime shape assumes one CoreML layer shard per model layer, so the manifest and Swift chaining logic must represent contiguous fused ranges without weakening ANE-only guarantees.

**Lesson**: Once fused shards pass residency and golden, the next throughput gain should come from making the runtime topology match the validated fused artifact topology.

**Next**: Implement fused manifest export and range-aware Swift runtime loading, compile, then run a bounded tok/s profile only; do not delete artifacts or run powermetrics.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="034-phi-4-mini-full-3-layer-shard-strategy-validated.html">Previous: Journal 034</a> | <a href="036-phi-4-mini-fused-runtime-migration-outcome.html">Next: Journal 036</a></nav>
