---
layout: default
title: "Journal 112 - INT4 Palettize L0 FFN Probe: All Gates Pass"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="111-o2-concurrent-ffn-partial-fan-out.html">Previous: Journal 111</a></nav>

# 2026-05-14 - INT4 Palettize L0 FFN Probe: All Gates Pass

**Intent**: Test whether INT4 per-grouped-channel palettization (`constexpr_lut_to_dense`, nbits=4, k-means, group_size=32) lands on ANE and meets the 0.97 cosine quality gate. This path is explicitly distinct from the previously-failed linear INT4 per-block path (`constexpr_blockwise_shift_scale`), which causes GPU fallback. The distinction is critical: LUT palettization bakes cluster centroids into the model at export time; block-wise shift-scale relies on runtime dequant that the ANE compiler cannot fuse. Prior failure documented in docs/INT4_SHARD_ANE_BUG.md; new path documented in [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md). Motivation: 75% compression vs FP16 baseline would cut the ~250 MB shard limit impact and reduce external scratch storage storage for the 30-layer × 8-shard matrix. Reference: the validation-first notes (validate representative sample before scale-out).

**Setup**: Scope: L0 FFN only (8 shards: p0–p6 + p7/last+combiner). Quant: `constexpr_lut_to_dense`, nbits=4, k-means, group_size=32 (`coremltools.optimize.coreml.palettize_weights`). Residency check: `MLComputePlan.load_from_path`, `CPU_AND_NE` target. Quality check: 5 unit-norm random seeds vs FP16 reference, shard p0of8, cosine similarity. Env: Xcode `python3` (coremltools 9 only — not `.venv` or `.venv313`). Artifacts: `external scratch storage:<external-scratch>/local model weights` with suffix `_q4_pal`.

**Result**: 8/8 L0 FFN shards compiled. Sizes: p0–p6 = 46 MB each, p7 (last+combiner) = 54 MB. Baseline comparison: FP16 = 182 MB (p0–p6) / 216 MB (p7). Compression: ~75%. ANE residency (`MLComputePlan`, CPU_AND_NE): 34 real compute ops → ANE, GPU=0, CPU=0. UNK=48 = const (44) + `ios18.constexpr_lut_to_dense` (4) — compile-time ops, no runtime device assignment. Quality: cosine 0.985 mean vs FP16 reference across 5 seeds (all ≥ 0.97). Two seeds produced zero output from both FP16 and palettize — confirmed routing behaviour (pack 0 has no active experts for those inputs, not a quant bug). Gates: ANE residency PASS, cosine quality PASS. Scale-out to all 30 layers unblocked. Scale-out running at time of entry (~2.3 h estimated, 30 layers × 8 shards × ~35 s/shard).

**Surprise / hurdle**: The `ios18.constexpr_lut_to_dense` ops appearing as UNK in `MLComputePlan` initially looked like an ANE fallback. Confirmed they are compile-time constant-folding ops with no runtime device assignment — not a residency failure. The two zero-output seeds from both FP16 and palettize were also initially suspicious; root cause is pack-0 expert routing (no active experts for those tokens), not a quantization artefact.

**Lesson**: `constexpr_lut_to_dense` INT4 palettization is a viable ANE-resident path; the previously-documented GPU fallback risk (INT4 shard bug) is specific to `constexpr_blockwise_shift_scale` (linear per-block) and does not apply to LUT palettization.

**Next**: Await scale-out completion (~30 layers × 8 shards). Then run full-stack end-to-end quality gate vs helper script (cosine ≥ 0.97 at model level). If end-to-end gate passes, INT4pal becomes the new production baseline for Gemma shards, replacing FP16 and INT8 per-tensor.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="111-o2-concurrent-ffn-partial-fan-out.html">Previous: Journal 111</a></nav>
