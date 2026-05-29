---
layout: default
title: "Experiment 20 - Weighted-Automaton Layer Partition Search"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="19-balanced-tree-reduction-semigroup-accumulator.html">Previous: Experiment 19</a> | <a href="21-apl-style-token-stream-batching.html">Next: Experiment 21</a></nav>

# Experiment 20 - Weighted-Automaton Layer Partition Search

**Sources**: Sakarovitch weighted automata + Dragon Book global optimization

Model layer topology as a shortest-path problem:

- states: layer indices `0..32`
- edges: existing or candidate compiled shards `[i,j)`
- invalid edges: CPU fallback, failed golden, known NaN, or missing artifact
- edge weight: measured `ms/token` from `ProfileDecodeLayers`

The first tool for this is `python/phi4_mini_topology_search.py`. It scans
existing `.mlmodelc` artifacts, profile logs, residency reports, and golden
reports, then reports both:

- the best observed whole-profile topology, avoiding cross-run timing mixing
- an edge-min lower bound, useful as a hint but not a benchmark claim

Initial result: `20+4+6+2` is the current best observed public topology
(`17.203 tok/s`, `53.039 ms/token` in layers), while `[0,24)` remains rejected
as a compiler cliff and `[24,32)` remains rejected for golden NaNs.

First follow-up: `20+5+5+2` was legal but slower. `[20,25)` and `[25,30)`
both passed ANE residency and golden (`cos=0.999350` and `0.999258`), but the
profile landed at `17.043 tok/s` and `53.565 ms/token` in layers. The tail is
therefore not just a shard-count problem; the `[20,24)+[24,30)` split remains
the better compiler/resource shape.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="19-balanced-tree-reduction-semigroup-accumulator.html">Previous: Experiment 19</a> | <a href="21-apl-style-token-stream-batching.html">Next: Experiment 21</a></nav>
