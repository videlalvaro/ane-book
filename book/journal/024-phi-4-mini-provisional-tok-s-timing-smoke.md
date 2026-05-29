---
layout: default
title: "Journal 024 - Phi-4-mini Provisional Tok/s Timing Smoke"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="023-phi-4-mini-tok-s-timing-smoke-intent.html">Previous: Journal 023</a> | <a href="025-phi-4-mini-lean-ane-runtime-optimization-intent.html">Next: Journal 025</a></nav>

# 2026-04-27 - Phi-4-mini Provisional Tok/s Timing Smoke

**Intent**: Measure current prompt-ID decode throughput after runtime scaffolding, following measurement-before-optimization discipline.

**Setup**: Added lightweight timing lines to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift), recompiled local artifacts, and ran bounded prompt-ID smokes with completed Phi-4-mini layer shards plus 4 LM-head shards.

**Result**: Provisional timing smoke results: max-new 8 generated 8 tokens, prefill 18.440395s, decode 7 tokens in 0.919999s = 7.609 tok/s, forward 8 calls in 19.360394s = 0.413 tok/s; max-new 16 prefill 18.596279s, decode 15 in 2.186287s = 6.861 tok/s, forward 16 in 20.782566s = 0.770 tok/s; max-new 32 prefill 18.832114s, decode 31 in 4.538147s = 6.831 tok/s, forward 32 in 23.370261s = 1.369 tok/s; max-new 64 prefill 18.796948s, decode 63 in 9.190524s = 6.855 tok/s, forward 64 in 27.987472s = 2.287 tok/s. Current warm decode throughput is about 6.8–6.9 tok/s; first token includes about 18.7s CoreML/model/state warmup.

**Surprise / hurdle**: The first-token path is dominated by CoreML/model/state warmup, so whole-run tok/s is misleading for short generations.

**Lesson**: Report warm decode tok/s separately from first-token warmup until the runtime has tokenizer and full-logit validation gates.

**Next**: Treat these as provisional smoke numbers only; no energy/powermetrics, tokenizer integration, HF full-logit golden validation, cleanup, or deletion was performed.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="023-phi-4-mini-tok-s-timing-smoke-intent.html">Previous: Journal 023</a> | <a href="025-phi-4-mini-lean-ane-runtime-optimization-intent.html">Next: Journal 025</a></nav>
