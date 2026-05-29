---
layout: default
title: "Journal 103 - Hy-MT1.5 2-bit GGUF ANE Conversion Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="102-phi-4-mini-rope96-fast-fused-rebuild-outcome.html">Previous: Journal 102</a> | <a href="104-speculative-decode-prompt-density-validation.html">Next: Journal 104</a></nav>

# 2026-04-30 - Hy-MT1.5 2-bit GGUF ANE Conversion Intent

**Intent**: Before any expensive conversion, record the intent to convert the Hugging Face model `AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF` to CoreML shards that run 100% on Apple Neural Engine in this repo. The base model is `tencent/HY-MT1.5-1.8B`; Hugging Face reports architecture `hunyuan-dense`, about 1.8B parameters, and a 574MB 2-bit GGUF. The plan follows the ANE-only mandate and validation-before-scale discipline: prove architecture support, compiler placement, and golden quality before treating any compression format as usable.

**Setup**: Planning note only; no model download, conversion, compilation, residency check, golden validation, latency run, energy benchmark, cleanup, or deletion has been run for this entry. Source quantization is 2-bit GGUF, but target production shard baseline remains INT8 per-tensor CoreML unless a smaller representative alternative passes ANE residency and golden quality gates. Linear INT4 per-block remains known risky for sharded Conv/Linear graphs; 2-bit GGUF compression is not accepted as proof of ANE residency.

**Result**: Intent and constraints recorded before the expensive run. No artifacts, placement counts, cosine/RMSE, perplexity, latency, energy, or compiled-size numbers exist yet for this model.

**Surprise / hurdle**: Architecture support is unknown: the repo converters must first be inspected for `hunyuan-dense` support. The normal flow calls for `optimality-gatekeeper`, but this session's available agent list does not include it, so the main agent should proceed conservatively with small analysis/probes and avoid a full long conversion if architecture support is absent.

**Lesson**: A small compressed GGUF is only a source artifact; ANE acceptance starts at converter support plus strict MLComputePlan residency and golden quality, not at the GGUF bit width.

**Next**: Inspect repo converter support for `hunyuan-dense`; download or identify the GGUF only if the path is plausible; run analyze/plan on the smallest representative shape if supported; then convert through strict ANE residency and golden gates before any scale-out or performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="102-phi-4-mini-rope96-fast-fused-rebuild-outcome.html">Previous: Journal 102</a> | <a href="104-speculative-decode-prompt-density-validation.html">Next: Journal 104</a></nav>
