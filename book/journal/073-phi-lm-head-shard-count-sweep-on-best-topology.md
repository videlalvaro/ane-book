---
layout: default
title: "Journal 073 - Phi LM-Head Shard Count Sweep on Best Topology"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="072-phi-4-mini-partial-rope-root-cause-intent.html">Previous: Journal 072</a> | <a href="074-phi-long-decode-topology-baseline-moves-to-20-4-6-2.html">Next: Journal 074</a></nav>

# 2026-04-28 - Phi LM-Head Shard Count Sweep on Best Topology

**Intent**: Test whether changing LM-head shard count improves the remaining `~5 ms/token` LM-head wall time after private E5 chaining proved low leverage for the current layer topology.

**Setup**: Generated comparable runtime manifests for the same `16+8+6+2` layer stack with 3-way and 8-way LM-head shards, reusing existing compiled artifacts. Ran strict `MLComputePlan` residency on every 3-way and 8-way LM-head shard before accepting the benchmark comparison; all passed with `compute_non_ane=0`. Profile command shape matched the 4-way baseline: 5 warmup calls, 30 generated tokens, `--profile`.

**Result**: 4-way remains best. 3-way: `16.695 tok/s`, `head_predict_reduce_ms=5.136`. 4-way rerun: `17.171 tok/s`, `head_predict_reduce_ms=5.095`. 8-way: `16.740 tok/s`, `head_predict_reduce_ms=5.156`.

**Surprise / hurdle**: More shards increase aggregate shard work without reducing head wall time; fewer shards reduce aggregate work but still do not improve wall time. This is not a simple parallelism knob.

**Lesson**: The current LM-head bottleneck is likely fixed CoreML/ANE submission plus reduction/scheduling overhead around the shards, not just per-shard matmul size. Keep the 4-way LM-head as the measured baseline.

**Next**: Look for algorithmic LM-head reductions that preserve ANE-only heavy compute, such as hierarchical shortlist/projection schemes with golden quality gates, rather than more shard-count reshuffling.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="072-phi-4-mini-partial-rope-root-cause-intent.html">Previous: Journal 072</a> | <a href="074-phi-long-decode-topology-baseline-moves-to-20-4-6-2.html">Next: Journal 074</a></nav>
