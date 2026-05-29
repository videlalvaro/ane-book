---
layout: default
title: "Experiment 25 - Prompt-Lookup Force Mode as a Head-Skip Ceiling"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="24-structured-cot-as-a-grammar-constrained-sampler.html">Previous: Experiment 24</a> | <a href="26-multi-token-verifier-feasibility.html">Next: Experiment 26</a></nav>

# Experiment 25 - Prompt-Lookup Force Mode as a Head-Skip Ceiling

**Sources**: Knuth pattern matching + Dechter constraint propagation + public
CoreML `MLState.withMultiArray(for:)` state access

Public CoreML state access is better than expected: Python exposes
`MLState.read_state`/`write_state`, and the Swift SDK exposes
`MLState.withMultiArray(for:)`. This means exact state copy is possible without
unsupported runtime path. It does not by itself create speculative speedup, because copying
the full Phi KV cache is a large host memory transfer and a single-token verifier
still performs one ANE layer pass per target token. Real pass-count speedup still
needs batch-token layer artifacts or another way to verify multiple tokens per
ANE call.

To quantify the cheap public ceiling, `phi4_mini_ane.swift` now has an
experimental approximate mode:

```bash
--ngram-force --ngram-min 2 --ngram-max 8
```

Unlike `--ngram-probe`, this changes generation: if prompt lookup finds a prior
suffix match, the runtime forces the proposed token and skips the ANE LM-head
prediction/reduction for that step. The ANE layer stack still runs so KV state
stays aligned with the emitted token stream.

Code-shaped suite result on the same 5 prompts / 95 decode tokens:

| mode | decode tokens | decode seconds | weighted tok/s | avg layer ms | avg head ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact greedy + `--ngram-probe` | 95 | 5.605536 | 16.948 | 53.876 | 5.120 |
| approximate `--ngram-force` | 95 | 5.269287 | 18.029 | 54.755 | 0.703 |

`--ngram-force` forced 82 of 100 target opportunities (`force_rate=0.820`) and
reduced mean head time by about 4.4 ms/token, but total speed improved only
`~6.4%` because the layer stack is now dominant. This is useful as a ceiling
measurement and maybe as a workload-specific approximate mode, but it is not an
exact speculative decoder and should not be the default shipping path without a
task-quality gate.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="24-structured-cot-as-a-grammar-constrained-sampler.html">Previous: Experiment 24</a> | <a href="26-multi-token-verifier-feasibility.html">Next: Experiment 26</a></nav>
