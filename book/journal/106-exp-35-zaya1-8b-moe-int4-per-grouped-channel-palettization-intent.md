---
layout: default
title: "Journal 106 - Exp 35: ZAYA1-8B MoE INT4 Per-Grouped-Channel Palettization Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="105-exp-26-follow-up-prompt-length-sweep-for-n-gram-speculative-decode.html">Previous: Journal 105</a> | <a href="107-exp-35-complete-zaya1-8b-int4pal-t-1-win-speculative-decode-loss.html">Next: Journal 107</a></nav>

# 2026-05-13 - Exp 35: ZAYA1-8B MoE INT4 Per-Grouped-Channel Palettization Intent

**Intent**: Replace the INT8 per-tensor quantized ZAYA1-8B MoE shards (Exp 34, 202 MB compiled each) with INT4 per-grouped-channel palettized shards (`constexpr_lut_to_dense`, `group_size=32`) to halve shard size (~101 MB target) and halve per-token FFN compute. Hypothesis: the halved verifier cost (~250–300 ms/call vs. current ~500 ms) will push the speculative decode break-even acceptance rate below the real n-gram acceptance rate on code prompts (~60–80%), yielding net throughput improvement over the 8.59 tok/s INT8 baseline. The LUT palettization approach maps 32-channel groups to 4-bit indices into a 16-entry codebook — analogous to Iverson APL §2 inner-product reduction over a finite alphabet — which is architecturally distinct from the linear INT4 per-block path (`constexpr_blockwise_shift_scale`) that is known to cause CPU fallback on small sharded MoE graphs.

**Setup**: ZAYA1-8B (Zyphra, 80-layer MoE, alternating attn/MoE layers, 40 MoE shards). Target quant: `constexpr_lut_to_dense` palettization, `group_size=32`, 4-bit per grouped channel. Baseline artifact: INT8 per-tensor MoE shards at 202 MB compiled each (temporary output). Env: Xcode `python3` / coremltools 9. Gate sequence (per ANE_CHAIN_SCHEMA.md): (1) ane-validator on L01 INT4 palettized shard — must be 100% ANE, no CPU fallback on any matmul/norm; (2) golden-validator — cosine ≥ 0.97 vs. temporary output; only after both gates pass: build all 40 shards + benchmark.

**Result**: Intent recorded. No artifacts produced yet; no ANE residency numbers, shard sizes, latency, energy, cosine, or perplexity measured.

**Surprise / hurdle**: The key risk is whether `constexpr_lut_to_dense` (LUT palettization) routes cleanly to ANE on ZAYA's MoE FFN shapes. It has been validated on Gemma-4 shards (T4.1.0-1.1) but not yet on ZAYA. The per-block INT4 path (`constexpr_blockwise_shift_scale`) is confirmed-bad on small sharded graphs; this experiment is on the separate palettization path and must not be conflated with that known failure. The ane-validator gate on L01 specifically targets this risk before any scale-out work.

**Lesson**: Validated palettization on one model family (Gemma-4) is not transferable to another (ZAYA MoE) without re-running the ANE residency gate; never skip the single-shard gate before committing 40-shard conversion work.

**Next**: Run ane-validator on L01 palettized shard; if 100% ANE, run golden-validator; if cosine ≥ 0.97, build all 40 MoE shards and benchmark decode tok/s vs. 8.59 tok/s INT8 baseline. If ANE residency fails, diagnose which ops fall back and decide between shape-tuning or abandoning this quant path.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="105-exp-26-follow-up-prompt-length-sweep-for-n-gram-speculative-decode.html">Previous: Journal 105</a> | <a href="107-exp-35-complete-zaya1-8b-int4pal-t-1-win-speculative-decode-loss.html">Next: Journal 107</a></nav>
