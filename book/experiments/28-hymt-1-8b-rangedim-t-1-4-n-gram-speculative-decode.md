---
layout: default
title: "Experiment 28 - HyMT 1.8B RangeDim T=1..4 + N-Gram Speculative Decode"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="27-microgpt-on-ane-minimum-size-constraint-discovery.html">Previous: Experiment 27</a> | <a href="29-zaya1-8b-moe-feasibility-probe-on-ane.html">Next: Experiment 29</a></nav>

# Experiment 28 - HyMT 1.8B RangeDim T=1..4 + N-Gram Speculative Decode

**Date**: 2025-05-12

**Sources**: APL/Iverson (Notation as a Tool of Thought): dynamic array semantics
drive `ct.RangeDim` — a single compiled program handles any T in [1,4] at runtime.
Dragon Book §9.2 (data-flow analysis): the T-agnostic `HeadRMSNorm` is a classic
loop-hoisting transformation — the reshape over `n_heads` is folded into the static
channel axis so no T-dependent control flow remains in the traced graph.

**Context**: Port of the Phi-4-mini RangeDim + speculative decode pipeline (Exp 26)
to HyMT 1.8B (Hunyuan Dense, d=2048, 32L, GQA 16/4, has_qk_norm=True, vocab=120818,
max_seq_len=512, INT8 per-tensor, tied embeddings).

**HyMT-specific challenge — T-agnostic per-head QK norm**:
HyMT applies RMSNorm independently to each of 16 Q heads and 16 K heads after
QKV projection. Naïve reshape `[1, d_model, T, 1] → [n_heads, d_head]` would be
T-dependent. Fix (Iverson §2 on rank-polymorphism):

```python
chunks = x.chunk(n_heads, dim=1)   # split static channel axis
# each chunk: [1, d_head, T, 1] — T is left in the spatial dim, untouched
mean_sq = chunk.pow(2).mean(dim=1, keepdim=True)  # [1, 1, T, 1] — T-agnostic
norm = chunk * (mean_sq + eps).rsqrt() * weight_tiled
```

`x.chunk(n_heads, dim=1)` cuts the static channel (dim=1) into n_heads groups of
`[1, d_head, T, 1]`; the RMS mean over dim=1 is independent of T. This pattern
is T-agnostic at trace time, giving `ct.RangeDim` freedom to JIT-specialize T at
runtime without retracing.

**Shard topology**:
7 shards: 6×(5 layers, ~241.8 MB compiled) + 1×(2 layers, ~96.7 MB compiled).
All 7 pass `conv_non_ane=0` residency check. LM head: 2× T=1 INT8 shards covering
vocab [0,60409) and [60409,120818).

**Parity validation**:
| Comparison | Cosine similarity |
|-----------|-------------------|
| Old T=1 shard vs new RangeDim (T=1) | **1.000000** (bit-exact) |
| RangeDim T=1 vs T=4 (slot 0) | **1.000000** (bit-exact) |

**Benchmark** (M4 Max, `--prompt-ids 120000 --max-new 50`):

| Mode | Decode tok/s | Speedup |
|------|-------------|---------|
| Baseline T=1 | 37.2 | 1× |
| Speculative `--speculative` | **60.3** | **+62%** |

The repeating-token test (BOS → BOS×50) is the best-case for n-gram speculation
(bigram accepted at every step). Real-world gain will track the acceptance-rate
formula from Exp 23 and Exp 26.

**Artifacts**:
- `python/hymt_rangedim_export_shard.py` — export script (HeadRMSNorm, RangeDim T=1..4)
- `local-artifacts/hymt_ane_rangedim/` — 7 compiled `.mlmodelc` shards
- `local-artifacts/hymt_ane/hymt_runtime_meta_rangedim.json` — runtime manifest
- `local-artifacts/hymt_ane/lm_head_shards/HymtLMHead_s{0,1}_q8.mlmodelc` — LM head shards
- `local-artifacts/hymt_ane.swift` — speculative decode runtime (ported from phi4)
- `python/hymt_rangedim_parity_check.py` — parity check script (cosine validation)

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="27-microgpt-on-ane-minimum-size-constraint-discovery.html">Previous: Experiment 27</a> | <a href="29-zaya1-8b-moe-feasibility-probe-on-ane.html">Next: Experiment 29</a></nav>
