---
layout: default
title: "Journal 042 - Phi-4-mini Isolated Warm Cache Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="041-phi-4-mini-full-4-layer-fused-strategy-completed.html">Previous: Journal 041</a> | <a href="043-ane-internals-synthesis-before-phi-daemon.html">Next: Journal 043</a></nav>

# 2026-04-27 - Phi-4-mini Isolated Warm Cache Outcome

**Intent**: Add explicit isolated warm-cache support to the Phi-4-mini Swift runtime so agent-session startup latency is paid before real generation, following the validation-first notes measurement-before-optimization and call-hoisting discipline.

**Setup**: Updated [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) with `--warmup-calls N` and `--warmup-token-id ID`. Warmup runs with separate `MLState`s for layer shards, then resets attention/KV write masks so the real generation KV cache starts clean. Tested on the 4-layer fused manifest.

**Result**: With `--warmup-calls 1`, cold first predict moved into warmup: warmup elapsed 97.192795s, real prefill_s=0.126592, decode_tok_s=14.563. With `--warmup-calls 4`: warmup elapsed 99.890267s, real prefill_s=0.083317, decode_tok_s=14.598, forward_tok_s=14.573. Current bottleneck remains layer chain about 63.4 ms/token plus LM-head fanout/reduce about 5.14 ms/token.

**Surprise / hurdle**: Deeper warmup slightly improved real prefill but did not materially improve steady decode; for dense Phi there is no token-dependent MoE router, so routing optimization means fixed layer-shard scheduling plus LM-head shard fanout.

**Lesson**: Explicit isolated warm cache fixes agent-session first-token latency after startup, but steady Phi-4-mini decode is still dominated by ANE layer-chain time and LM-head fanout/reduce.

**Next**: No energy benchmark was run; next optimization should target layer-chain latency or LM-head fanout/reduce while preserving ANE-only compute and clean KV-cache semantics.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="041-phi-4-mini-full-4-layer-fused-strategy-completed.html">Previous: Journal 041</a> | <a href="043-ane-internals-synthesis-before-phi-daemon.html">Next: Journal 043</a></nav>
