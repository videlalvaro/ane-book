---
layout: default
title: "Journal 071 - Phi Unsupported Stream-Level One-Stream Timing Reality Check"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="070-phi-full-fused-topology-runs-in-one-stream.html">Previous: Journal 070</a> | <a href="072-phi-4-mini-partial-rope-root-cause-intent.html">Next: Journal 072</a></nav>

# 2026-04-28 - Phi Unsupported Stream-Level One-Stream Timing Reality Check

**Intent**: Measure whether the validated unsupported one-stream stream-level path materially improves Phi decode latency by removing public CoreML hidden-state roundtrips between fused layer shards.

**Setup**: Added `--iterations` and `--warmup-iterations` to `local two-stage stream probe --manual-chain-all`. Ran the full `16+8+6+2` fused layer stack with 10 warmup executes and 100 measured executes. Re-ran the public Swift runtime on `phi4mini_runtime_meta_16_8_6_2.json` with 5 warmup calls, 30 generated tokens, and `--profile`.

**Result**: The unsupported stream stayed correct, with final hidden sum `-196.834778`. Unsupported one-stream layers measured `52.593 ms/execute`; public CoreML layers measured `53.121 ms/token`. Public decode was `17.179 tok/s`, with `head_predict_reduce_ms=5.082`.

**Surprise / hurdle**: The unsupported stream win is real but small: about `0.53 ms/token` for this already-fused topology. The host hidden-state roundtrip is not the primary bottleneck once the topology is `16+8+6+2`.

**Lesson**: Unsupported Stream-Level chaining is a capability breakthrough, not an immediate large throughput breakthrough for the current Phi topology. It may matter more for finer sharding, but current speed work should focus on ANE compute shape/topology and LM-head latency.

**Next**: Keep the unsupported chain as a validated research path; prioritize higher-leverage public/ANE optimizations unless a future topology needs many more shard boundaries.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="070-phi-full-fused-topology-runs-in-one-stream.html">Previous: Journal 070</a> | <a href="072-phi-4-mini-partial-rope-root-cause-intent.html">Next: Journal 072</a></nav>
