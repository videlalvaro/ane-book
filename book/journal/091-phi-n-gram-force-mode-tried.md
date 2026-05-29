---
layout: default
title: "Journal 091 - Phi N-Gram Force Mode Tried"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="090-phi-structured-cot-runtime-slice-implemented.html">Previous: Journal 090</a> | <a href="092-phi-multi-token-verifier-feasibility-insight.html">Next: Journal 092</a></nav>

# 2026-04-29 - Phi N-Gram Force Mode Tried

**Intent**: Try the public n-gram idea as an actual runtime speed experiment, not just an acceptance probe.

**Setup**: First checked public CoreML state access. Python `MLState` exposes `read_state`/`write_state`; Swift exposes `withMultiArray(for:)`. State copy is therefore possible, but copying the full Phi KV cache plus single-token verification would not reduce ANE target passes. Added experimental `--ngram-force` to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) instead.

**Result**: `--ngram-force` trusts prompt-lookup proposals, forces those token IDs, and skips LM-head prediction/reduction for forced steps while still running the ANE layer stack. This is approximate and changes generation, unlike `--ngram-probe`.

**Validation**: Swift compile passed. Rebuilt local artifacts. Regenerated temporary output and benchmarked exact `--ngram-probe` versus approximate `--ngram-force` on the same 5 prompts with `max-new=24`.

**Numbers**: Exact greedy/probe: 95 decode tokens, 5.605536 decode seconds, weighted `16.948 tok/s`, avg `layers_ms=53.876`, avg `head_ms=5.120`, accepted 69/100 probe targets. Approx force: 95 decode tokens, 5.269287 decode seconds, weighted `18.029 tok/s`, avg `layers_ms=54.755`, avg `head_ms=0.703`, forced 82/100 opportunities.

**Lesson**: Prompt-lookup head skipping gives only a `~6.4%` throughput win because the layer stack dominates. The big `2x+` speculative upper bound still requires batch-token verifier artifacts or another multi-token ANE verification shape.

**Next**: Keep `--ngram-force` experimental and off by default. Do not ship it as a correctness path until coding-task quality says approximate prompt lookup is acceptable.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="090-phi-structured-cot-runtime-slice-implemented.html">Previous: Journal 090</a> | <a href="092-phi-multi-token-verifier-feasibility-insight.html">Next: Journal 092</a></nav>
