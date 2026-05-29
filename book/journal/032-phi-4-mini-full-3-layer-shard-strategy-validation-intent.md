---
layout: default
title: "Journal 032 - Phi-4-mini Full 3-Layer Shard Strategy Validation Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="031-phi-4-mini-three-layer-full-shard-probe-passed.html">Previous: Journal 031</a> | <a href="033-llm-int8-ane-conv2d-adaptation-review-intent.html">Next: Journal 033</a></nav>

# 2026-04-27 - Phi-4-mini Full 3-Layer Shard Strategy Validation Intent

**Intent**: After the layers 0–3 3-layer Phi-4-mini probe passed and the user said “ok, validate that,” validate the full 3-layer shard strategy across the whole 32-layer model, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Planned non-destructive output directory: local artifacts. Ranges: [0,3), [3,6), [6,9), [9,12), [12,15), [15,18), [18,21), [21,24), [24,27), [27,30), and tail [30,32). For each range, compile if missing, then run strict MLComputePlan residency and range golden smoke.

**Result**: Intent recorded before execution; no new range artifact counts, placement, cosine, latency, energy, or perplexity numbers yet.

**Surprise / hurdle**: The first 288M 3-layer compile passed despite exceeding the older conservative shard-size caution, so every range must re-prove compile success, ANE residency, and numerical smoke instead of assuming scale-out safety.

**Lesson**: A fused-shard strategy is validated only when every planned range passes the same residency and quality gates, including the shorter tail.

**Next**: Run the full range validation only in the non-destructive probe directory; do not clean up/delete artifacts and do not run performance or energy benchmarking.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="031-phi-4-mini-three-layer-full-shard-probe-passed.html">Previous: Journal 031</a> | <a href="033-llm-int8-ane-conv2d-adaptation-review-intent.html">Next: Journal 033</a></nav>
