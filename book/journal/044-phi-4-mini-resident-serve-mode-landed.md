---
layout: default
title: "Journal 044 - Phi-4-mini Resident Serve Mode Landed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="043-ane-internals-synthesis-before-phi-daemon.html">Previous: Journal 043</a> | <a href="045-phi-4-mini-five-layer-fused-strategy-intent.html">Next: Journal 045</a></nav>

# 2026-04-28 - Phi-4-mini Resident Serve Mode Landed

**Intent**: Implement the first resident Phi-4-mini runtime service mode after saving the ANE-internals synthesis, treating CoreML execution as a loaded-artifact lifecycle and following measurement-before-optimization discipline.

**Setup**: Updated [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift) with `--serve`, which loads models once, optionally runs isolated warmup, then starts a JSON-lines protocol. Request schema: `{"prompt_ids":[...],"max_new":N,"profile":true}`. Responses include `ok`, `generated_ids`, `timing`, and optional `profile`. Serve mode keeps `MLModel` instances resident, creates fresh `MLState`s per request to avoid KV-cache leakage, resets masks per request, writes status logs to stderr, and reserves stdout for READY/JSON responses.

**Result**: Swift compile passed. One-shot warm smoke preserved behavior with `--warmup-calls 1 --max-new 2 --profile`: warmup=102.083947s, real prefill=0.145252s, decode_tok_s=15.353 for one decode token. Serve-mode two-request smoke passed in one process after warmup=100.955364s; both requests generated `[6360,198]`; request1 prefill=0.127760s and decode_tok_s=15.052; request2 prefill=0.084319s and decode_tok_s=14.822. READY handshake was also verified after recompilation.

**Surprise / hurdle**: Service correctness depended on separating resident model lifetime from per-request state lifetime, plus keeping stdout machine-readable while moving logs to stderr.

**Lesson**: A resident ANE service can amortize CoreML load/warmup while preserving clean KV-cache semantics by recreating `MLState`s and masks for every request.

**Next**: No energy benchmark was run; next steps are energy measurement and longer multi-request soak tests while preserving ANE-only compute and per-request cache isolation.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="043-ane-internals-synthesis-before-phi-daemon.html">Previous: Journal 043</a> | <a href="045-phi-4-mini-five-layer-fused-strategy-intent.html">Next: Journal 045</a></nav>
