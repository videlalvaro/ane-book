---
layout: default
title: "Journal 002 - Phi-4-mini-instruct ANE Support Scaffolding Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="001-fp16-gemma-ane-rebuild-retry-requested.html">Previous: Journal 001</a> | <a href="003-phi-4-mini-instruct-layer-0-gate-residency-passed.html">Next: Journal 003</a></nav>

# 2026-04-27 - Phi-4-mini-instruct ANE Support Scaffolding Intent

**Intent**: Start Phi-4-mini-instruct ANE support with safe scaffolding only: reusable analyzer/preflight and orchestration scripts before any expensive conversion. The plan follows the ANE-only mandate, quality-before-performance gating, and optimization discipline from the validation-first notes (measure and validate before scaling an implementation).

**Setup**: Workspace: `this repo`; model seed artifact: the local Phi-4-mini GGUF weights; planned baseline: INT8 per-tensor CoreML shards targeting ANE. Initial implementation scope is non-destructive preflight/analyzer/orchestration code only, with disk/RAM/cache guardrails and no full conversion, no cleanup of model/output artifacts, and no benchmarking.

**Result**: Intent recorded before implementation. No artifacts produced yet; no residency, latency, energy, cosine, or perplexity numbers yet.

**Surprise / hurdle**: Phi-4 support must be structured so that scaffolding cannot accidentally trigger heavyweight conversion or destructive cleanup while still encoding mandatory gates.

**Lesson**: New model support should begin with guardrailed orchestration that makes ANE residency and golden quality gates unavoidable before any performance work.

**Next**: Implement the analyzer/preflight and orchestration scripts; require MLComputePlan residency validation plus golden quality validation before any benchmark or scale-out conversion.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="001-fp16-gemma-ane-rebuild-retry-requested.html">Previous: Journal 001</a> | <a href="003-phi-4-mini-instruct-layer-0-gate-residency-passed.html">Next: Journal 003</a></nav>
