---
layout: default
title: "Journal 076 - Phi Weighted Topology Search Starts"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="075-phi-stream-timing-on-20-4-6-2.html">Previous: Journal 075</a> | <a href="077-phi-20-5-5-2-tail-probe.html">Next: Journal 077</a></nav>

# 2026-04-28 - Phi Weighted Topology Search Starts

**Intent**: Start moving from hand-picked fused layer topologies to a book-shaped search process for ANE-efficient computation shapes.

**Setup**: Added the topology search script, applying Sakarovitch weighted-automaton framing and Dragon Book compiler-cliff discipline. The script scans existing Phi `.mlmodelc` layer-range artifacts, Swift profile logs, residency JSON, and golden JSON. It treats layer indices as states and compiled shards as weighted edges, with known rejected edges `[0,24)` and `[24,32)` excluded.

**Result**: Initial scan found 72 existing compiled edges and 9 profile logs. It correctly reports `20+4+6+2` as the best whole observed profile (`layers_ms=53.039`, `decode_tok_s=17.203`) while separately showing an optimistic edge-min lower bound (`16+8+6+2`, `52.934 ms`) that mixes timings across runs and should not be treated as a benchmark claim.

**Surprise / hurdle**: The first DP pass exposed a measurement gotcha: per-edge minimum timings across different runs can beat any actually observed full topology. The tool now separates whole-profile winners from edge-min hints.

**Lesson**: Layer-shape optimization should be a graph search with explicit gates and whole-profile measurements, not a sequence of intuition-driven partitions.

**Next**: Use the searcher to choose candidate missing gates and future compiled ranges around the 20-layer cliff, then add a separate batch/token-shape probe for Iverson-style array work.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="075-phi-stream-timing-on-20-4-6-2.html">Previous: Journal 075</a> | <a href="077-phi-20-5-5-2-tail-probe.html">Next: Journal 077</a></nav>
