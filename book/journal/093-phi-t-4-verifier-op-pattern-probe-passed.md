---
layout: default
title: "Journal 093 - Phi T=4 Verifier Op-Pattern Probe Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="092-phi-multi-token-verifier-feasibility-insight.html">Previous: Journal 092</a> | <a href="094-phi-4-mini-real-weight-t-4-verifier-intent.html">Next: Journal 094</a></nav>

# 2026-04-29 - Phi T=4 Verifier Op-Pattern Probe Passed

**Intent**: Test the riskiest public multi-token verifier compiler pattern before spending disk/RAM on real Phi block artifacts.

**Setup**: Added helper script, a synthetic CoreML probe that builds a stateful T=4 transformer-like block with multi-row KV write, causal block attention, FFN, INT8 weight quantization, CoreML compilation, numerical block-vs-sequential check, and `MLComputePlan` residency report. Artifacts are under temporary output.

**Result**: Tiny shape `d=64` failed residency (`conv_non_ane=4`, `compute_non_ane=97`), confirming that very small graphs are not representative. Medium shape `d=1024 nh=16 nkv=4 dh=64 dff=2048 S=256 T=4` passed with `coreml_seq_vs_block_cos=0.999974`, `conv_non_ane=0`, `compute_non_ane=0`. Phi-sized synthetic shape `d=3072 nh=24 nkv=8 dh=128 dff=8192 S=512 T=4` also passed with `coreml_seq_vs_block_cos=0.999997`, `rmse=0.000322`, `conv_total=4/4 ANE`, `compute_total=145/145 ANE`.

**Surprise / hurdle**: The multi-row state ops stayed on ANE at Phi dimensions: `read_state`, `slice_update`, `write_state`, `softmax`, `matmul`, and all convs preferred ANE. The non-representative tiny shape falling to CPU is a cost-model warning, not a rejection of the verifier pattern.

**Lesson**: The T=4 KV scatter/update op family is viable on ANE at Phi dimensions. The next risk is real-weight conversion/golden parity, not compiler placement of the synthetic op pattern.

**Next**: Add real Phi `--batch-tokens 4` layer conversion for a one-layer block verifier, then compare against four sequential single-token calls before scaling to the `20+4+6+2` verifier topology.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="092-phi-multi-token-verifier-feasibility-insight.html">Previous: Journal 092</a> | <a href="094-phi-4-mini-real-weight-t-4-verifier-intent.html">Next: Journal 094</a></nav>
