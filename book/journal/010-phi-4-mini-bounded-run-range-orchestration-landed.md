---
layout: default
title: "Journal 010 - Phi-4-mini Bounded Run-Range Orchestration Landed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="009-phi-4-mini-bounded-run-range-orchestration-intent.html">Previous: Journal 009</a> | <a href="011-phi-4-mini-layers-4-7-bounded-run-range-intent.html">Next: Journal 011</a></nav>

# 2026-04-27 - Phi-4-mini Bounded Run-Range Orchestration Landed

**Intent**: Record the outcome of adding bounded Phi-4-mini layer-range orchestration while preserving the ANE-only, quality-before-scale workflow and validation discipline.

**Setup**: the Phi orchestration script now has `run-range` with `--layer-start` inclusive, `--layer-end` exclusive, default safety cap `--max-range-layers 4`, preflight before each layer, skip of convert/compile when the compiled artifact already exists, and strict residency plus numerical smoke per layer. Validation commands: `bash -n the Phi orchestration script`; dry-run `run-range --layer-start 4 --layer-end 6 --gatekeeper-go --dry-run`.

**Result**: `bash -n the Phi orchestration script` passed. The dry-run generated the expected layer 4 and layer 5 commands without executing heavy work.

**Surprise / hurdle**: The range semantics and safety cap needed to make automation convenient without silently expanding into full scale-out.

**Lesson**: Bounded range orchestration is safest when every layer re-enters preflight and must pass residency plus numerical smoke before progressing.

**Next**: No conversion, compile, perf, energy, cleanup, or deletion was run by this validation; the next real range should remain capped and gated.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="009-phi-4-mini-bounded-run-range-orchestration-intent.html">Previous: Journal 009</a> | <a href="011-phi-4-mini-layers-4-7-bounded-run-range-intent.html">Next: Journal 011</a></nav>
