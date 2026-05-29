---
layout: default
title: "Experiment 23 - Prompt-Lookup / N-Gram Speculation"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="22-hierarchical-lm-head-reduction.html">Previous: Experiment 22</a> | <a href="24-structured-cot-as-a-grammar-constrained-sampler.html">Next: Experiment 24</a></nav>

# Experiment 23 - Prompt-Lookup / N-Gram Speculation

**Sources**: Knuth string matching + Concrete Mathematics amortization

Prompt lookup can predict repeated continuations by matching the current token
suffix against prior context and proposing the token that followed the previous
match. This is attractive for coding-agent workloads because generated code,
logs, diffs, stack traces, and structured outputs often contain long local
repetitions.

The first implementation is intentionally measurement-only:

```
local-artifacts/phi4_mini_ane_runtime --ngram-probe --ngram-min 2 --ngram-max 8
```

It runs exact greedy decode unchanged, then reports how often a prompt-lookup
n-gram proposal would have existed and matched the model's actual next token.
The initial repetitive smoke run produced:

```
targets=30 proposals=24 accepted=24 proposal_rate=0.800 acceptance_rate=1.000
```

This proves proposal signal exists, but it is not yet a speedup. With the
current stateful single-token layer shards, skipping a proposed token would leave
the KV cache incomplete, and verifying several proposed tokens ahead would need
a public rollback/copy strategy for `MLState` or separate batch-token layer
artifacts. Until that verifier exists, n-gram speculation is a probe and design
input, not a production acceleration path.

The next probe added `--prompt-ids-file` and a reproducible prompt generator,
`python/phi4_mini_ngram_prompt_suite.py`, so several code-shaped token prompts
can reuse one loaded runtime. The initial 5-prompt code suite measured:

```
NGramProbeSuite: targets=100 proposals=74 accepted=69
proposal_rate=0.740 acceptance_rate=0.932 accepted_per_target=0.690
```

That acceptance density is high enough to justify a public verifier design, but
the same KV-cache caveat remains: exact speedup requires block verification or
state rollback, not just proposal prediction.

`python/phi4_mini_ngram_spec_sim.py` replays runtime logs and estimates the
verifier-pass upper bound if a public block verifier can check several proposed
tokens at once. On the same code suite:

```
draft=4: generated=100 verifier_passes=49 ideal_speedup=2.04x
draft=8: generated=100 verifier_passes=41 ideal_speedup=2.44x
```

These are target-pass counts, not measured runtime throughput. Real speedup
depends on building batch-token layer artifacts that keep KV updates exact and
ANE-resident.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="22-hierarchical-lm-head-reduction.html">Previous: Experiment 22</a> | <a href="24-structured-cot-as-a-grammar-constrained-sampler.html">Next: Experiment 24</a></nav>
