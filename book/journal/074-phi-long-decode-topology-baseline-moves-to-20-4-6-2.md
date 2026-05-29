---
layout: default
title: "Journal 074 - Phi Long-Decode Topology Baseline Moves to 20+4+6+2"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="073-phi-lm-head-shard-count-sweep-on-best-topology.html">Previous: Journal 073</a> | <a href="075-phi-private-e5-timing-on-20-4-6-2.html">Next: Journal 075</a></nav>

# 2026-04-28 - Phi Long-Decode Topology Baseline Moves to 20+4+6+2

**Intent**: Re-check existing fused layer topologies after private E5 timing showed boundary removal was low leverage and the layer stack remained the dominant cost.

**Setup**: Profiled existing public CoreML manifests with the same 4-way LM head, 10 warmup calls, 100 generated tokens, and `--profile`. Ran strict residency on the `20+4+6+2` layer shards and numerical range golden gates for `[0,20)` and `[20,24)`.

**Result**: `20+4+6+2` is the new measured long-decode best: `17.203 tok/s`, `layers_ms=53.039`, `head_predict_reduce_ms=5.084`. The `16+8+6+2` rerun over 100 generated tokens measured `16.596 tok/s`, `layers_ms=55.084`. Gates passed for `20+4+6+2`: `[0,20)` residency `compute_non_ane=0`, golden `cos_hidden=0.998546`; `[20,24)` residency `compute_non_ane=0`, golden `cos_hidden=0.999446`; tail shards were resident in the same check and previously validated.

**Surprise / hurdle**: The earlier short-run `16+8+6+2` winner is not the best long-decode point. Larger front fusion to 20 layers reduces layer overhead enough to win, while `[0,24)` remains beyond the compiler residency cliff.

**Lesson**: Fused topology should be selected on long decode profiles, not only short bursts; `20+4+6+2` is the current public baseline under ANE-only gates.

**Next**: Use `phi4mini_runtime_meta_20_4_6_2.json` for further public-runtime comparisons, and keep `[0,24)` rejected unless compiler behavior changes.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="073-phi-lm-head-shard-count-sweep-on-best-topology.html">Previous: Journal 073</a> | <a href="075-phi-private-e5-timing-on-20-4-6-2.html">Next: Journal 075</a></nav>
