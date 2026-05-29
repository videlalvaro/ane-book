---
layout: default
title: "Journal 026 - Phi-4-mini Lean ANE Runtime Optimization Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="025-phi-4-mini-lean-ane-runtime-optimization-intent.html">Previous: Journal 025</a> | <a href="027-phi-4-mini-decode-profile-intent.html">Next: Journal 027</a></nav>

# 2026-04-27 - Phi-4-mini Lean ANE Runtime Optimization Outcome

**Intent**: Optimize [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) for a leaner Phi-4-mini ANE runtime while preserving ANE-only heavy compute. Techniques followed the validation-first notes: Dragon Book allocation/strength reduction by hoisting and reusing `MLDictionaryFeatureProvider` and `MLFeatureValue` allocations outside the per-layer hot loop; Iverson/APL whole-array partitioning by treating the 4 LM-head shards as independent vocab slices; and Stepanov-style reduction by reducing 4 local argmaxes to one global argmax.

**Setup**: Swift prompt-ID runtime with 32 ANE layer shards and 4 ANE LM-head shards. Host work remained limited to embedding lookup, RoPE/mask bookkeeping, and argmax/reduction. Per-token stdout was removed from the default hot path behind `--trace`. Compiled optimized runtime successfully with no diagnostics.

**Result**: Quiet optimized timing: max-new64 exact run: prefill 18.939262s, decode 63 tokens in 7.810305s = 8.066 tok/s, forward 64 calls in 26.749567s = 2.393 tok/s. max-new128: prefill 19.095862s, decode 127 tokens in 15.949655s = 7.963 tok/s, forward 128 in 35.045517s = 3.652 tok/s. Compared to the original 64-token decode baseline of 6.855 tok/s, the best 64-token run improved +17.7%; 128-token sustained decode improved +16.2%.

**Surprise / hurdle**: Removing hot-path stdout and CoreML provider/value allocation churn was enough to expose a material decode-throughput gain without moving any heavy compute back to CPU/GPU.

**Lesson**: ANE-resident graphs still need lean host orchestration; allocation hoisting, shard partitioning, and small reductions can improve sustained tok/s while preserving the ANE-only compute boundary.

**Next**: No powermetrics/energy benchmark, cleanup, or deletion was run; next step is a separate energy measurement and any further runtime changes should keep layer and LM-head compute on ANE.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="025-phi-4-mini-lean-ane-runtime-optimization-intent.html">Previous: Journal 025</a> | <a href="027-phi-4-mini-decode-profile-intent.html">Next: Journal 027</a></nav>
