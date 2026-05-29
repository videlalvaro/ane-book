---
layout: default
title: "Journal 015 - Phi-4-mini Run-Batches Landed and Layers 8–11 Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="014-phi-4-mini-layers-8-11-run-batches-intent.html">Previous: Journal 014</a> | <a href="016-phi-4-mini-layers-12-15-run-batches-intent.html">Next: Journal 016</a></nav>

# 2026-04-27 - Phi-4-mini Run-Batches Landed and Layers 8–11 Passed

**Intent**: Record the `run-batches` orchestration outcome and the first bounded actual batch, following validation-before-scale discipline and the project ANE-only gate policy.

**Setup**: the Phi orchestration script now has a future-facing `run-batches` stage with `--batch-size` default 4, explicit `--layer-end` required to avoid silent full conversion, `--stop-after-batches`, resource preflight before each batch, and delegation to `run-range` for per-layer convert, compile, strict residency, and golden gates. Validation dry-run: `run-batches --layer-start 8 --layer-end 16 --batch-size 4 --stop-after-batches 2 --gatekeeper-go --dry-run` produced batches 8–12 and 12–16. Actual bounded run: `run-batches --layer-start 8 --layer-end 12 --batch-size 4 --stop-after-batches 1 --gatekeeper-go`.

**Result**: PASS. Layers 8–11 converted and compiled successfully. Strict residency passed for each layer: conv_total=4 conv_ane=4 conv_non_ane=0; compute_total=148 compute_ane=148 compute_non_ane=0. Golden passed: L8 cos=0.999868 rmse=0.014178 max_abs=0.085938; L9 cos=0.999853 rmse=0.015281 max_abs=0.056641; L10 cos=0.999875 rmse=0.016508 max_abs=0.109375; L11 cos=0.999807 rmse=0.016104 max_abs=0.093750.

**Surprise / hurdle**: The batch runner stayed bounded while requiring an explicit end layer and reusing `run-range` gates rather than duplicating per-layer logic.

**Lesson**: Batch automation is safe when it is explicit, stopped, preflighted per batch, and delegates every layer to the same residency and golden gates.

**Next**: No full 32-layer conversion, performance run, energy run, cleanup, or deletion was run; continue only through explicit gated bounded batches.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="014-phi-4-mini-layers-8-11-run-batches-intent.html">Previous: Journal 014</a> | <a href="016-phi-4-mini-layers-12-15-run-batches-intent.html">Next: Journal 016</a></nav>
