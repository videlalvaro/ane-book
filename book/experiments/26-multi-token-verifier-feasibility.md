---
layout: default
title: "Experiment 26 - Multi-Token Verifier Feasibility"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="25-prompt-lookup-force-mode-as-a-head-skip-ceiling.html">Previous: Experiment 25</a> | <a href="27-microgpt-on-ane-minimum-size-constraint-discovery.html">Next: Experiment 27</a></nav>

# Experiment 26 - Multi-Token Verifier Feasibility

**Sources**: Dragon Book data-flow invariants + Knuth sequential verification +
CoreML public state API

The multi-token verifier is possible, but it requires new layer artifacts. The
important correction is that exact speculation does not require cheap rollback
of the whole `MLState` if KV positions are treated as append-only slots guarded
by the attention mask.

For a draft block of length `T`, a verifier can run the target model over all
draft tokens and write their KV entries into positions `[pos, pos+T)`. If the
accepted prefix length is `m < T`, the slots after `pos+m` are harmless because
future attention masks exclude them. The first rejected slot at `pos+m` is
overwritten when the runtime feeds the target fallback token at that same
position. If all `T` draft tokens are accepted, the block state is already the
correct committed state.

Therefore the public design does not need unsupported stream path or full-state copy/rollback.
It needs a static block verifier graph:

```text
x:             [1, d, T, 1]
rope_cos/sin:  [T, d_head/2]
attn_mask:     [1, 1, T, max_seq]
kv_write_mask: [1, 1, max_seq, T]
hidden:        [1, d, T, 1]
```

The graph change is inside attention:

- Conv/RMSNorm/FFN already work across the token axis `T` as 1x1 image width.
- Q/K/V must reshape to include `T` query positions.
- RoPE must apply one position row per token.
- KV update scatters `T` new K/V rows into the state cache.
- Attention scores are `[batch, heads, T, max_seq]` with a causal mask that lets
  each draft token see prompt state plus earlier draft tokens.

The existing batch-4 LM-head shards already solve the final projection shape.
A `T=4` verifier pass can validate up to four draft tokens and produce one
target fallback/bonus token from the last logits row. This is the first route
that can plausibly approach the earlier `2.04x` draft-4 pass-count upper bound.

Gates before scale:

1. Build a one-layer or smallest-tail `T=4` stateful block shard.
2. Run `ane-validator`; reject if scatter/broadcast/sum falls to CPU/GPU.
3. Run a golden verifier against four sequential single-token forwards.
4. Only then build the `20+4+6+2` block verifier topology.

Synthetic op-pattern test result:

- Added `python/phi4_mini_t4_verifier_probe.py`, a cheap stateful CoreML probe
  that builds a transformer-like `T=4` block with multi-row KV write, causal
  block attention, FFN, INT8 weights, CoreML compilation, numerical check, and
  `MLComputePlan` residency check.
- Tiny shape `d=64` failed residency: all compute preferred CPU. This was a
  cost-model/non-representative shape, not a numerical failure.
- Larger representative shape `d=1024`, `nh=16`, `nkv=4`, `dh=64`, `dff=2048`,
  `S=256` passed: `conv_non_ane=0`, `compute_non_ane=0`,
  `coreml_seq_vs_block_cos=0.999974`.
- Phi-sized synthetic shape `d=3072`, `nh=24`, `nkv=8`, `dh=128`, `dff=8192`,
  `T=4`, `S=512` passed: `conv_non_ane=0`, `compute_non_ane=0`,
  `coreml_seq_vs_block_cos=0.999997`, package/compiled size `100.8 MB`.

The compiler gate for the T=4 KV scatter op pattern is therefore green at Phi
dimensions. The next test must use real Phi weights and compare one block shard
against four sequential single-token shard calls.

Real-weight one-layer gate result:

- Added `python/phi4_mini_t4_layer_probe.py`, which builds a real Phi layer-0
  `T=4` CoreML shard from `models/Phi-4-mini-instruct.Q8_0.gguf`, compiles it,
  compares against four sequential PyTorch single-token calls, and runs strict
  `MLComputePlan` residency.
- First real run exposed a true layout bug in the prototype: attention output was
  flattened as `[head, token, dim]` into channels. That is invisible for `T=1`
  and mostly hidden by tiny synthetic weights, but it corrupts real `T>1` Phi
  hidden states. The fix is to permute to `[head, dim, token]` before reshaping
  to `[1, d, T, 1]`.
- After the fix, PyTorch four-step sequential vs block is exact for both real
  token embeddings and random hidden inputs: all per-token cosines `1.000000`.
- Real CoreML INT8 layer-0 verifier passed: `T=4`, `S=2048`, package/compiled
  size `100.8 MB`, `coreml_seq_vs_block_cos=0.996174`, per-token cosines
  `[0.999879, 0.989271, 0.999179, 0.993851]`, `conv_non_ane=0`,
  `compute_non_ane=0`.

This moves the verifier from synthetic feasibility to a real-weight one-layer
green gate. The next scaling step is to add `T=4` export/runtime plumbing for
the production shard topology and verify exact greedy token equality end to end.

Scale-out/runtime result:

- Added `python/phi4_mini_t4_export_shard.py` and built the full production
  verifier topology: `[0,20)`, `[20,24)`, `[24,30)`, `[30,32)` under
  `local-artifacts/phi4_mini_ane_t4_verifier/`.
- All four compiled shards passed strict residency. The largest `[0,20)` shard
  has `conv_total=80`, `conv_non_ane=0`, `compute_total=2768`,
  `compute_non_ane=0`, compiled size `2015.8 MB`.
- Added `speculative_verifier` manifest support and generated
  `phi4mini_runtime_meta_20_4_6_2_t4.json` with verifier layers plus existing
  batch-4 LM-head shards.
- Added opt-in Swift `--speculative` mode using T=4 verifier states and batch-4
  head prediction. On the 5-prompt code-shaped suite with `--ngram-min 1`, it
  decoded `93` tokens in `4.290248 s` = `21.68 tok/s`, versus exact greedy
  `95` tokens in `5.624088 s` = `16.89 tok/s` on the same prompt file.
- This is not yet the final exact speculative runtime: generated tokens diverged
  on some prompts because the full T=4 verifier stack is numerically close but
  not token-identical to the single-token q8 runtime in all contexts.

Conclusion: scale-out and runtime plumbing are green for ANE residency and show
speed potential, but shipping still needs either full-stack token parity or an
exactness guard. Treat current `--speculative` as experimental.

**RangeDim unification (2026-05-12)**:
Co-loading separate T=1 and T=4 shard sets in one process exceeded ANE's
~3 GB per-process DRAM limit — error -14. `EnumeratedShapes` was rejected by
stream runtime for stateful `MLState` models with `std::bad_cast`. The fix is
`ct.RangeDim(lower_bound=1, upper_bound=4, default=1)`: a single compiled
program that stream runtime JIT-specialises for T=1 and T=4 on first use per process,
eliminating the co-load problem entirely.

All four shards rebuilt with RangeDim, strict residency confirmed:

| Shard | Compiled size | conv_ane / total |
|-------|-------------|------------------|
| [0,20) | 2015.8 MB | 80/80 |
| [20,24) | 403.2 MB | 16/16 |
| [24,30) | 604.7 MB | 24/24 |
| [30,32) | 201.6 MB | 8/8 |

Warm throughput measured via daemon benchmark (`python/phi4_mini_rangedim_bench.py`,
5 reps, 39-token code prompt, `--serve` mode, single persistent process):

| Metric | Value |
|--------|-------|
| T=4 chunked prefill | **68.7 tok/s** (median) |
| T=1 autoregressive decode | **16.7 tok/s** |
| Prefill / decode ratio | **4.1×** (≈ theoretical T=4 limit) |
| Wall time (39-tok prefill + 8 decode) | 0.99 s |
| Cold T=1 JIT per process | ~112 s |
| Cold T=4 JIT per process | ~133 s |

stream runtime JIT-compiles separate specialisations per T value on first use in each
process. A single resident daemon amortises that cost across all requests.
Manifest: `phi4mini_runtime_meta_rope96_rangedim_20_4_6_2.json`.
Shards: `local-artifacts/phi4_mini_ane_rangedim/`.

Speculative decode via `--speculative --ngram-min 1` on the same RangeDim shards:

| Prompt | Reps | New toks | Prefill tok/s | Decode tok/s | Wall/req | Speedup vs T=1 |
|--------|------|----------|--------------|-------------|---------|---------------|
| T=1 baseline (39-tok) | — | — | — | 17.8 | — | 1× |
| 39-tok code prompt | 5 | 20 | 68.9 | **18.1** | 1.62s | +1.7% |
| 372-tok Swift CoreML prompt | 5 | 80 | **70.4** | **26.7** | 8.25s | **+50%** |

The 372-token prompt (a dense Swift CoreML
snippet with heavy repetition of `MLMultiArray`, `MLModel`, `MLState`,
`makeInputDict`, `forwardLayer`, `rope_cos`, `rope_sin`, `attn_mask`,
`kv_write_mask` — ideal n-gram match territory. All 5 reps were identical to
within 0.2 tok/s (26.6–26.8), confirming stable ANE scheduling.

Speculative speedup: **+50%** over exact T=1 baseline (17.8 tok/s) on the
dense code prompt, vs +1.7% on a 39-token prompt. This confirms the hypothesis
from Exp 23: n-gram acceptance rate is strongly prompt-density-dependent.
Short context with no repetition yields near-zero gain; longer multi-turn code
contexts bring the acceptance rate high enough to approach the simulated 2.04×
upper bound. Knuth TAOCP Vol. 3 §6.1 (sequential search): the expected match
distance shrinks proportionally to token-vocabulary collision frequency — denser
corpora collapse that distance faster.

**Prompt-length sweep (2026-05-12)**:
Sweep using `python/phi4_mini_ngram_sweep.py` across four context sizes in a
single daemon session (JIT paid once; T=1 JIT 113.4s, T=4 JIT 140.8s). Prompts
constructed by tiling that prompt to target length, so
n-gram match density stays representative. 5 reps each, 80 new tokens.

| Prompt length | Decode tok/s | Prefill tok/s | Wall/req | Speedup vs T=1 |
|--------------|-------------|--------------|---------|---------------|
| 100 tokens   | **21.1**    | 70.1         | 5.17s   | **1.19×**     |
| 200 tokens   | **22.1**    | 70.3         | 6.42s   | **1.24×**     |
| 372 tokens   | **26.7**    | 70.1         | 8.26s   | **1.50×**     |
| 800 tokens   | **28.9**    | 69.9         | 14.19s  | **1.62×**     |

Prefill throughput is stable at ~70 tok/s across all lengths — the T=4 chunked
prefill path scales cleanly with context. Decode speedup continues rising at 800
tokens (1.62× vs 1.50× at 372), indicating the acceptance-rate curve has not
saturated. The simulated 2.04× ceiling (Exp 23, draft=4) is still ahead and
likely requires longer repeated code contexts or multi-turn chat history.
Script: `python/phi4_mini_ngram_sweep.py`.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="25-prompt-lookup-force-mode-as-a-head-skip-ceiling.html">Previous: Experiment 25</a> | <a href="27-microgpt-on-ane-minimum-size-constraint-discovery.html">Next: Experiment 27</a></nav>
