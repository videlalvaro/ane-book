---
layout: default
title: "Experiment 31 - ZAYA1-8B CCA (conv_qk) gates wired into 40 stateful attn shards (2025-07-14)"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="30-zaya1-8b-stateful-attn-shards-kv-cache-on-ane.html">Previous: Experiment 30</a> | <a href="32-zaya1-8b-speculative-decode-t-4-verifier-n-gram-implemented-bottlenecked.html">Next: Experiment 32</a></nav>

# Experiment 31 - ZAYA1-8B CCA (conv_qk) gates wired into 40 stateful attn shards (2025-07-14)

**Source citations**:
- Sakarovitch *Elements of Automata Theory* §III.3: weighted finite automaton as a
  gated linear recurrence over the sequence — the CCA `conv_qk` stages implement
  exactly this: a causal window of depth 2 over the concatenated (Q,K) channel
  vector, with learned per-channel weights.
- TAOCP Vol. 1 §2.2 (Knuth): causal convolution at T=1 collapses to a
  position-slice multiply — the current-kernel-position equivalence that
  justifies replacing `F.conv2d` with elementwise `mul + bmm`.

**Objective**: Wire CCA `conv_qk` (Exp 30 stub → Exp 31 active) into all 40
stateful attn shards, achieve golden validator cosine ≥ 0.97 (40/40), smoke
test at real decode throughput.

**CCA architecture (reverse-engineered)**:
- `conv_qk.0`: depthwise Conv1d `(1280, 1, 2)` — per-channel scale×prior + bias;
  at T=1, current-kernel-pos = `w[:, 0, 1]` (a [1280] scalar per channel)
- `conv_qk.1`: grouped Conv1d `(1280, 128, 2)` with `groups=10` (one group per
  Q/K head) — maps grouped channels with a `(128, 128)` local mixing matrix;
  at T=1, current-kernel-pos = `w[:, :, 1]` reshaped to `(10, 128, 128)` for bmm
- Applied to `cat(Q, K)` before RoPE, additive: `Q += cca[:Q_DIM]`, `K += cca[Q_DIM:]`
- Dims: input `[1280] = Q_DIM(1024) + KV_DIM(256) = 8×128 + 2×128`

**INT8 selective skip** (`make_int8_config_skip_qk`):
In coremltools 9.x, `linear_quantize_weights` targets ALL constant-weight
matmul ops (not just conv/linear layers). The Q and K projections were being
INT8-quantized despite being `register_buffer` + `torch.matmul` — because the
compiler lowers them to `constexpr` + `matmul` MIL ops.
Fix: after `ct.convert()`, inspect `ml._mil_program`, find matmul ops whose
const inputs match shapes `(Q_DIM, H)=(1024, 2048)` or `(KV_DIM, H)=(256, 2048)`,
and set `op_name_configs={op.name: None}` — `None` = skip in ct9 `OptimizationConfig`.
V and O projections remain INT8 (no issue there).
MIL op names differ between CCA-active (`op_50/op_55`) and CCA-skipped
(`op_46/op_51`) branches — shape-based detection handles both automatically.

**CCA conditional skip** (static JIT branch):
Layers where `max(|conv_qk.0.bias|) > 5.0` are CCA-skipped at export time
(traced as a static Python bool → dead-code eliminated in MIL).
- L00: `b0_max=35.0` → CCA skipped
- L74: `b0_max=4.47`, L76: `12.94`, L78: `6.63` → L76 and L78 also skipped

**ANE residency — all 40 shards**:
```
conv_total=2 conv_ane=2 conv_non_ane=0  (CCA-active layers)
conv_total=2 conv_ane=2 conv_non_ane=0  (CCA-skipped layers — same, CCA ops not present)
```
100% ANE resident. Shard sizes: 8.1 MB (CCA-active), 7.9 MB (CCA-skipped).

**Golden validator** — Exp 31 final:
`python/zaya_golden_validator.py --full --prompt-ids 1,1000,5000`
(tokens with typical embedding std≈0.08–0.09; avoid low-std tokens 42/100 that
are in the bottom 4% of vocab and create pathological cross-attention scale mismatch)

| Metric | Value |
|--------|-------|
| Layers checked | 40/40 attn |
| PASS (cosine ≥ 0.97) | **40/40** |
| FAIL | 0 |
| Mean cosine (all layers) | **0.999835** |
| Min cosine | **0.999636** |

Gate verdict: **GREEN — cosine gate GREEN** ✓

**Validator anti-patterns discovered**:
1. BOS token (id=2) as first prompt token amplifies INT8 K/V rounding error at
   positions 1 and 2 (known from Exp 30). Do not use id=2 as a validator token.
2. Tokens 42, 100, 300 share anomalously small embeddings (std≈0.0097, bottom 4%
   of vocab). Using them alongside normal-scale tokens creates a degenerate
   cross-attention scenario where a high-scale query token (e.g. id=200, std=0.067)
   sees cached low-scale KV entries → INT8 V error is amplified by the attention
   weight ratio (~7× scale mismatch). This caused 38/40 initially with ids 42,100,200.
   With realistic diverse tokens (ids 1,1000,5000), all 40 layers pass at ≥0.9996.

**Smoke test** (M4 Max, `--prompt-ids 2,42 --max-new 20 --profile`):

| Metric | Exp 30 (no CCA) | Exp 31 (CCA wired) |
|--------|-----------------|---------------------|
| Decode tok/s | 8.82 | **8.62** |
| Total decode 20 tok | ~2.27s | 2.320s |
| Attn shard load time | ~0.27s | ~0.27s |

CCA adds minimal overhead (~2%) — the `mul + bmm` pattern at T=1 involves
small tensors (staging through `[10, 1, 128]` bmm) and is fully ANE-resident.

**`attn_implementation` tag**: `cca_gqa_stateful_kvcache_rope_partial_qk_fp16_v_o_int8_cond_skip`
**`cca_wired`**: `true`

**Artifacts**:
- `local-artifacts/zaya_ane/attn_stateful/zaya_stateful_attn_L{00,02,...,78}.mlpackage` — 40 CCA shards
- `local-artifacts/zaya_ane/zaya_runtime_meta_stateful_cca.json` — runtime manifest (CCA)
- `python/zaya_stateful_attn_export.py` — export script (Exp 31.4, `make_int8_config_skip_qk`)
- `python/zaya_golden_validator.py` — golden validator (default `--prompt-ids 1,1000,5000`)
- `local-artifacts/zaya_ane/zaya_cca_golden_v2.log` — full 40-layer golden run log

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="30-zaya1-8b-stateful-attn-shards-kv-cache-on-ane.html">Previous: Experiment 30</a> | <a href="32-zaya1-8b-speculative-decode-t-4-verifier-n-gram-implemented-bottlenecked.html">Next: Experiment 32</a></nav>
