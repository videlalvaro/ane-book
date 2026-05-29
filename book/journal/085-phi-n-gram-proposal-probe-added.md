---
layout: default
title: "Journal 085 - Phi N-Gram Proposal Probe Added"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="084-phi-public-algorithmic-perf-direction-intent.html">Previous: Journal 084</a> | <a href="086-phi-code-shaped-n-gram-suite-measured.html">Next: Journal 086</a></nav>

# 2026-04-29 - Phi N-Gram Proposal Probe Added

**Intent**: Try a public n-gram/prompt-lookup direction without resorting to unsupported stream path APIs and without changing exact greedy decode behavior.

**Setup**: Added `--ngram-probe`, `--ngram-min`, and `--ngram-max` to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift). The probe runs normal greedy decode and records whether the current token history has a prior suffix match whose following token would have proposed the model's actual next token.

**Result**: Swift runtime rebuilt successfully. Smoke command on `phi4mini_runtime_meta_20_4_6_2.json` with `--ngram-min 2 --ngram-max 8` generated 30 targets and reported `proposals=24`, `accepted=24`, `proposal_rate=0.800`, `acceptance_rate=1.000`, `accepted_per_target=0.800`. Most accepted proposals came from N=8 on the repetitive output pattern.

**Surprise / hurdle**: Proposal quality can be high on repetitive output, but it is not yet a speedup. Current public CoreML Phi layer shards mutate `MLState` one token at a time; exact speculative block verification still needs rollback/copy semantics, batch-token layer artifacts, or another commit/discard mechanism.

**Lesson**: N-gram prompt lookup is worth pursuing as a public algorithmic path, but first as an acceptance-rate/workload-selection probe. It must not be used to skip ANE layer execution unless skipped tokens still populate the KV cache correctly.

**Next**: Run the probe on coding-like token streams and decide between two public follow-ups: batch-token layer verifier artifacts, or multi-stream batching where independent streams avoid rollback.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="084-phi-public-algorithmic-perf-direction-intent.html">Previous: Journal 084</a> | <a href="086-phi-code-shaped-n-gram-suite-measured.html">Next: Journal 086</a></nav>
