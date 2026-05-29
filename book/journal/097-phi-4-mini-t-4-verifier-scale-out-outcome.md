---
layout: default
title: "Journal 097 - Phi-4-mini T=4 Verifier Scale-Out Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="096-phi-4-mini-t-4-verifier-scale-out-intent.html">Previous: Journal 096</a> | <a href="098-phi-4-mini-t-4-speculative-exactness-comparison-intent.html">Next: Journal 098</a></nav>

# 2026-04-29 - Phi-4-mini T=4 Verifier Scale-Out Outcome

**Intent**: Record the full production-topology `T=4` verifier scale-out outcome for public CoreML speculative decoding, following Experiment 26, Knuth sequential verification, Dragon Book data-flow invariants, and Leviathan et al. (2023) speculative decoding. The goal was to test whether four-token verifier passes could preserve exact greedy behavior while improving decode throughput on the existing `20+4+6+2` Phi topology.

**Setup**: Built all four production-topology `T=4` verifier shards under local artifacts: `[0,20)`, `[20,24)`, `[24,30)`, and `[30,32)`. Added runtime manifest local artifacts with `speculative_verifier` entries and batch-4 LM-head references. Added an opt-in Swift `--speculative` path using separate `T=4` verifier states plus the batch-4 LM head. Suite command used `--ngram-min 1 --ngram-max 8`.

**Result**: All four verifier shards compiled and passed strict residency with `conv_non_ane=0` and `compute_non_ane=0`. Smoke results were mixed: a short odd prompt diverged at the final prefill position, while code prompt 0 matched exact greedy for 24 tokens. On the prompt suite, speculative weighted decode reached 93 tokens in 4.290248s = 21.68 tok/s versus exact greedy 95 tokens in 5.624088s = 16.89 tok/s. Outputs were not exact on all prompts because of full-stack verifier drift, so this is an experimental/approximate path, not the final exact speculative runtime.

**Surprise / hurdle**: ANE residency scaled cleanly across the production verifier topology, but full-stack exactness did not; the runtime can be faster while still failing the core speculative-decoding acceptance contract on some prompts.

**Lesson**: A multi-token verifier is useful only when exactness is gated as strongly as ANE residency; throughput wins without full-stack parity remain experimental.

**Next**: Solve full-stack verifier parity or add an exactness guard before shipping any speculative runtime; do not treat the current `--speculative` path as final exact decode despite its measured suite speedup.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="096-phi-4-mini-t-4-verifier-scale-out-intent.html">Previous: Journal 096</a> | <a href="098-phi-4-mini-t-4-speculative-exactness-comparison-intent.html">Next: Journal 098</a></nav>
