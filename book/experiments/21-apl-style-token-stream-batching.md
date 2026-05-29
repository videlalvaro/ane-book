---
layout: default
title: "Experiment 21 - APL-Style Token/Stream Batching"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="20-weighted-automaton-layer-partition-search.html">Previous: Experiment 20</a> | <a href="22-hierarchical-lm-head-reduction.html">Next: Experiment 22</a></nav>

# Experiment 21 - APL-Style Token/Stream Batching

**Sources**: Iverson APL inner/outer product + Concrete Mathematics amortization

Single-token decode is a poor ANE shape: `[1,D,1,1]` gives the conv engine only
one spatial point per weight load. The next array-shape probe should convert a
representative layer shard to accept `T > 1` positions, e.g. `[1,D,T,1]`, and
measure whether 1x1 conv weight reuse improves prefill, multi-agent serving, or
speculative verification.

This does not directly accelerate single-stream greedy decode unless speculation
or batching supplies independent tokens, but it can be the largest throughput
lever for coding-agent workloads.

First probe: the LM head now has an opt-in `--batch-tokens` builder path. The
full 4-shard `T=4` set, `hidden` shape `[1,3072,4,1]`, passed strict residency
(`conv_non_ane=0`, `compute_non_ane=0` on every shard) and numerical golden
against NumPy (`cos_logits` from `0.999926` to `0.999937`). A shard-0 microbench
measured one batched prediction at `0.691 ms/token` versus four single-token
predictions at `1.608 ms/token`, a `2.33x` per-token improvement for that shard.
This is a multi-stream/speculative/prefill shape lever, not a direct greedy
single-stream decode win until the runtime can supply independent hidden vectors.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="20-weighted-automaton-layer-partition-search.html">Previous: Experiment 20</a> | <a href="22-hierarchical-lm-head-reduction.html">Next: Experiment 22</a></nav>
