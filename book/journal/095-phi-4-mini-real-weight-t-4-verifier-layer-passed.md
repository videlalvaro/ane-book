---
layout: default
title: "Journal 095 - Phi-4-mini Real-Weight T=4 Verifier Layer Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="094-phi-4-mini-real-weight-t-4-verifier-intent.html">Previous: Journal 094</a> | <a href="096-phi-4-mini-t-4-verifier-scale-out-intent.html">Next: Journal 096</a></nav>

# 2026-04-29 - Phi-4-mini Real-Weight T=4 Verifier Layer Passed

**Intent**: Execute the next gate for the public max-performance speculative runtime: one real Phi-4-mini layer, four draft positions, exact sequential-vs-block parity, and strict ANE residency.

**Setup**: Added helper script. It builds layer 0 from the local Phi-4-mini GGUF weights, uses `T=4`, `S=2048`, real token embeddings `[199999, 200021, 14350, 200019]`, converts to CoreML INT8, compiles to `.mlmodelc`, runs compiled prediction, and writes reports under temporary output.

**Result**: Initial real run failed parity (`torch_seq_vs_block_cos=0.974411`) while still passing residency. Root cause was an attention-output layout bug: `[1, nh, T, dh]` was reshaped directly into `[1, d, T, 1]`, interleaving token positions into channels. Fixed by permuting to `[1, nh, dh, T]` before reshape. After the fix, PyTorch exactness passed for both embeddings and random hidden inputs (`torch_seq_vs_block_cos=1.000000`, all per-token cosines `1.000000`). CoreML INT8 real layer passed with `coreml_seq_vs_block_cos=0.996174`, per-token cosines `0.999879,0.989271,0.999179,0.993851`, `rmse=0.020813`, `max_abs=0.808594`.

**Residency**: Built artifact temporary output is fully ANE by both the integrated probe and standalone checker: `conv_total=4`, `conv_ane=4`, `conv_non_ane=0`, `compute_total=146`, `compute_ane=146`, `compute_non_ane=0`, `PASS=True`.

**Surprise / hurdle**: The synthetic probe was good enough for compiler placement but not enough to catch token/channel layout. Real weights made the bug obvious. This is a useful warning: all future `T>1` artifacts need per-token parity metrics, not just aggregate cosine.

**Lesson**: The public CoreML `T=4` verifier is now past both synthetic and real-weight one-layer gates. The next unknown is runtime integration and exact greedy equality across all 32 layers, not ANE residency of the core state/update pattern.

**Next**: Add `T=4` export plumbing for the production `20+4+6+2` topology, connect existing batch-4 LM-head shards, implement Swift speculative accept/reject, and prove exact token equality before any performance or energy claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="094-phi-4-mini-real-weight-t-4-verifier-intent.html">Previous: Journal 094</a> | <a href="096-phi-4-mini-t-4-verifier-scale-out-intent.html">Next: Journal 096</a></nav>
