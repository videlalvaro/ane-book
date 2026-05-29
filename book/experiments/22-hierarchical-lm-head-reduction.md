---
layout: default
title: "Experiment 22 - Hierarchical LM-Head Reduction"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="21-apl-style-token-stream-batching.html">Previous: Experiment 21</a> | <a href="23-prompt-lookup-n-gram-speculation.html">Next: Experiment 23</a></nav>

# Experiment 22 - Hierarchical LM-Head Reduction

**Sources**: Stepanov semigroup reduction + Iverson reduction operators

Flat LM-head argmax over 200k logits costs about `5 ms/token`; changing shard
count from 3 to 4 to 8 did not improve wall time. The next shape change is a
two-stage reduction:

1. ANE coarse projection or cluster scorer chooses a small candidate region.
2. ANE exact projection runs only on the shortlisted vocab rows.
3. CPU performs only trivial final argmax over a small returned set.

This must pass top-1/top-k agreement against the full LM head before any speed
claim. It is an algorithmic reduction-shape change, not a CPU shortcut.

Rejected shortcut: a CoreML `topk` LM-head shard was checked and failed the
ANE-only gate. The projection conv stayed on ANE, but `ios18.topk` and
`ios18.cast` executed on CPU, so this pattern must not be scaled.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="21-apl-style-token-stream-batching.html">Previous: Experiment 21</a> | <a href="23-prompt-lookup-n-gram-speculation.html">Next: Experiment 23</a></nav>
