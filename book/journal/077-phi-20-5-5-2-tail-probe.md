---
layout: default
title: "Journal 077 - Phi 20+5+5+2 Tail Probe"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="076-phi-weighted-topology-search-starts.html">Previous: Journal 076</a> | <a href="078-phi-4-mini-next-public-optimization-direction-intent.html">Next: Journal 078</a></nav>

# 2026-04-28 - Phi 20+5+5+2 Tail Probe

**Intent**: Test whether the public `20+4+6+2` baseline can be improved by using a more even post-20 tail split with already-built 5-layer shards.

**Setup**: Gated `phi4mini_layer20_25_q8.mlmodelc` and `phi4mini_layer25_30_q8.mlmodelc`, then generated `phi4mini_runtime_meta_20_5_5_2.json` and profiled 100 generation steps with 20 warmup calls.

**Result**: Both candidate shards passed residency (`conv_non_ane=0`, `compute_non_ane=0`) and golden (`[20,25)` cosine `0.999350`; `[25,30)` cosine `0.999258`). Runtime was slower than baseline: `17.043 tok/s`, `layers_ms=53.565`, with `L20-25=9.230ms` and `L25-30=9.096ms`.

**Lesson**: The post-20 tail does not prefer equal 5-layer tiling. The existing `[20,24)+[24,30)+[30,32)` split remains better, likely because the compiler/resource packing cost is nonlinear across layer positions and state shapes.

**Next**: Keep `20+4+6+2` as baseline and use the topology searcher for future candidates; the next larger lever is likely batching/token-shape or LM-head hierarchy rather than simple tail repartitioning.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="076-phi-weighted-topology-search-starts.html">Previous: Journal 076</a> | <a href="078-phi-4-mini-next-public-optimization-direction-intent.html">Next: Journal 078</a></nav>
