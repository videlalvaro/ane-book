---
layout: default
title: "Journal 023 - Phi-4-mini Tok/s Timing Smoke Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="022-phi-4-mini-runtime-scaffolding-smoke-passed.html">Previous: Journal 022</a> | <a href="024-phi-4-mini-provisional-tok-s-timing-smoke.html">Next: Journal 024</a></nav>

# 2026-04-27 - Phi-4-mini Tok/s Timing Smoke Intent

**Intent**: Answer the user's “what's our current tok/s?” question by adding lightweight timing to the Swift prompt-ID runtime and running a bounded throughput smoke, following measurement-before-optimization discipline.

**Setup**: Planned scope: instrument [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) around the existing full-chain prompt-ID decode path; run a small bounded `--max-new` tok/s smoke using completed Phi-4-mini layer shards and 4 LM-head shards. Constraints: no tokenizer integration, no HF full-logit golden validation, no powermetrics/energy run, and no cleanup or deletion.

**Result**: Intent recorded before implementation; tok/s, latency, energy, and validation numbers are pending.

**Surprise / hurdle**: Any reported throughput will be provisional because full HF golden logits validation and tokenizer integration are not complete.

**Lesson**: Throughput can be sampled only as a bounded smoke until quality and tokenizer gates make the runtime representative.

**Next**: Add minimal timing, compile the Swift runtime, run the bounded tok/s smoke, and record the provisional throughput with caveats.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="022-phi-4-mini-runtime-scaffolding-smoke-passed.html">Previous: Journal 022</a> | <a href="024-phi-4-mini-provisional-tok-s-timing-smoke.html">Next: Journal 024</a></nav>
