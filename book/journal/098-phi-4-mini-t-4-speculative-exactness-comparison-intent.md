---
layout: default
title: "Journal 098 - Phi-4-mini T=4 Speculative Exactness Comparison Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="097-phi-4-mini-t-4-verifier-scale-out-outcome.html">Previous: Journal 097</a> | <a href="099-phi-full-stack-gguf-reference-gate-blocks-q8-chat.html">Next: Journal 099</a></nav>

# 2026-04-29 - Phi-4-mini T=4 Speculative Exactness Comparison Intent

**Intent**: After committing tag `phi4-mini-t4-spec-runtime-exp-2026-04-29`, compare public-API Phi-4-mini speculative-runtime strategies because the current `T=4` path is faster on code-shaped prompts but not exact on all prompts. The comparison follows the ANE-only mandate, Experiment 26, Knuth-style sequential verification, and Leviathan et al. (2023) speculative decoding: throughput gains are useful only if exact greedy output is preserved or the deviation is explicitly diagnosed.

**Setup**: Planned strategies: (1) the currently implemented `T=4`-only speculative runtime; (2) hybrid exact-prefill plus `T=4` speculative decode; and possibly an exact-check diagnostic mode that measures `T=4` agreement against the canonical single-token exact path. Scope is public CoreML/runtime APIs on the existing Phi-4-mini ANE artifacts and `T=4` verifier topology. Heavy compute remains on ANE; no CPU/GPU compute fallback and no destructive artifact cleanup are planned.

**Result**: Intent recorded before the comparison. No new placement, exactness, latency, energy, cosine, perplexity, or throughput numbers yet beyond the prior observation that the current `T=4` suite was faster on code-shaped prompts but not exact on all prompts.

**Surprise / hurdle**: The measured failure mode is not ANE residency but full-stack exactness; the faster runtime cannot be accepted until the divergence source is isolated or avoided.

**Lesson**: Speculative decoding for the ANE path must be judged first by exact greedy parity and only then by tok/s, even when the verifier topology is fully ANE-resident.

**Next**: Run side-by-side prompt-suite comparisons for `T=4`-only versus hybrid exact-prefill plus `T=4` decode, add the exact-check diagnostic if needed, and record exact-match rates, first-divergence positions, and throughput deltas without deleting artifacts.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="097-phi-4-mini-t-4-verifier-scale-out-outcome.html">Previous: Journal 097</a> | <a href="099-phi-full-stack-gguf-reference-gate-blocks-q8-chat.html">Next: Journal 099</a></nav>
