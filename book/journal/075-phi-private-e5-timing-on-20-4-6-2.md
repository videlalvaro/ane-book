---
layout: default
title: "Journal 075 - Phi Private E5 Timing on 20+4+6+2"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="074-phi-long-decode-topology-baseline-moves-to-20-4-6-2.html">Previous: Journal 074</a> | <a href="076-phi-weighted-topology-search-starts.html">Next: Journal 076</a></nav>

# 2026-04-28 - Phi Private E5 Timing on 20+4+6+2

**Intent**: Re-test private one-stream E5 chaining on the newly promoted `20+4+6+2` public baseline.

**Setup**: Ran `e5_two_op_stream_probe --manual-chain-all --raw-bind-memory` on `[0,20) -> [20,24) -> [24,30) -> [30,32)` with 10 warmup executes and 100 measured executes. Ran the generalized public sequential probe on the same shard list for correctness.

**Result**: Public sequential and private one-stream match at every stage: sums `4568.3968`, `4590.55386`, `4822.20798`, final `-196.949768`. Private one-stream layers measured `51.662 ms/execute`, compared with public runtime layer calls at `53.039 ms/token` for the same topology.

**Surprise / hurdle**: The private boundary win is larger on `20+4+6+2` than on `16+8+6+2`, about `1.38 ms/token`, but still not a massive jump.

**Lesson**: Private E5 chaining can plausibly lift the new baseline from `17.203 tok/s` to roughly `17.6 tok/s` if integrated cleanly; useful, but still secondary to layer compute and LM-head algorithmic work.

**Next**: Productize private E5 only if that extra `~0.4 tok/s` matters enough to justify private API complexity; otherwise keep optimizing public ANE topology and LM-head strategy.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="074-phi-long-decode-topology-baseline-moves-to-20-4-6-2.html">Previous: Journal 074</a> | <a href="076-phi-weighted-topology-search-starts.html">Next: Journal 076</a></nav>
