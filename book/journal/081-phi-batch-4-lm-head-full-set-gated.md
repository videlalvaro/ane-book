---
layout: default
title: "Journal 081 - Phi Batch-4 LM-Head Full Set Gated"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="080-phi-batch-4-lm-head-shape-probe-passed.html">Previous: Journal 080</a> | <a href="082-phi-dead-artifact-cleanup-approval-intent.html">Next: Journal 082</a></nav>

# 2026-04-28 - Phi Batch-4 LM-Head Full Set Gated

**Intent**: Scale the passed representative batch-4 LM-head shape to the remaining vocab shards after checking disk headroom and before any runtime integration.

**Setup**: Built shards 1-3 into local artifacts, preserving shard 0. Free disk fell from `9.1 GiB` to `6.2 GiB`; no cleanup or deletion was performed.

**Result**: All four batch-4 LM-head shards now exist. Shards 1-3 passed strict residency with `conv_total=1`, `conv_ane=1`, `conv_non_ane=0`, `compute_total=8`, `compute_ane=8`, `compute_non_ane=0`, `PASS=True`. Goldens passed: shard 1 `cos_logits=0.999932`, shard 2 `0.999935`, shard 3 `0.999937`.

**Lesson**: The batch-token LM-head shape scales across the full Phi vocab shard set without the CPU fallback that killed the CoreML `topk` path.

**Next**: Keep artifacts local and metadata tracked. The next implementation step is a workload-specific runtime path for multi-stream/speculative/batched head scoring, not replacing the greedy single-token head path blindly.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="080-phi-batch-4-lm-head-shape-probe-passed.html">Previous: Journal 080</a> | <a href="082-phi-dead-artifact-cleanup-approval-intent.html">Next: Journal 082</a></nav>
