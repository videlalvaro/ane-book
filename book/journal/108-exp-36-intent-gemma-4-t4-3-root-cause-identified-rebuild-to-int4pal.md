---
layout: default
title: "Journal 108 - Exp 36 Intent: Gemma 4 T4.3 Root Cause Identified, Rebuild to INT4pal"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="107-exp-35-complete-zaya1-8b-int4pal-t-1-win-speculative-decode-loss.html">Previous: Journal 107</a> | <a href="109-t4-3-closed-all-fp16-ane-inference-passes-golden-gate.html">Next: Journal 109</a></nav>

# 2026-05-13 - Exp 36 Intent: Gemma 4 T4.3 Root Cause Identified, Rebuild to INT4pal

**Intent**: After exhausting ZAYA speculative decode, pivot to the unresolved Gemma 4 ANE full-stack quality failure (T4.3: per-position logit cosine dropped to 0.5654 at pos 2 for the 8-token golden prompt). All 90 Gemma ANE shards deleted (source GGUF intact); weights reside on the unmounted T9 volume. The goal is to identify the root cause without new probes and decide the rebuild quantization strategy for all 30 layers.

**Setup**: Root cause analysis via code inspection of the T4.1.3 rebuild artifacts and the [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md) INT4 vs. INT8 documentation. Single-layer quality validation data from T4.1.3: `cos(hidden)` range 0.9555–0.9999 across 7 sampled layers. Full-stack failure: 8-token golden prompt failed (cos = 0.5654 at pos 2); 6-token REAP prompt passed (cos ≥ 0.9875). MoE expert bank weight shape: 45056×704 (31M params per bank).

**Result**: Root cause identified (code analysis, no new hardware probes): T4.1.3 used **INT8 per-tensor quantization** for all 30 layers. Per-tensor quantization of large MoE expert banks uses ONE global scale per 31M-param tensor. Outlier weights in the expert banks force the scale high, leaving most weights at low effective precision. Some layers (likely mid-range FFN-heavy layers) show only 0.9555 cosine fidelity per-layer. Over 30 layers, cumulative error compounds into the observed T4.3 full-stack failure. Decision: Rebuild all 30 layers with **INT4 per-block palettized** (`constexpr_lut_to_dense`, per grouped channel) — the T4.1.2 approach, which gave cos(hidden)=0.992 on a real-weights single-layer test and is architecturally distinct from the linear INT4 per-block path known to cause CPU fallback. Requires mounting T9 or re-downloading 48 GB Gemma weights to external scratch storage. This is Exp 36.

**Surprise / hurdle**: The T4.1.3 per-tensor INT8 choice was made to avoid the "INT4 per-block linear fallback risk" documented in [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md) — but that risk applies specifically to `constexpr_blockwise_shift_scale` (linear INT4 per-block). The palettized path (`constexpr_lut_to_dense`) is separate and has now been validated on ZAYA (100% ANE residency, 40 shards). The irony: switching to INT8 per-tensor to avoid the INT4 CPU fallback risk introduced a quality regression that INT4 palettized does not have. Per-tensor INT8 is a worse choice than INT4 palettized for large MoE expert banks.

**Lesson**: For large MoE expert weight banks (>10M params per tensor), per-tensor INT8 quantization is precision-limited by outlier weights and can produce lower per-layer fidelity than INT4 grouped palettization; always validate per-layer cosine on a sample of layers before committing to full-model scale-out.

**Next**: Mount T9 (or re-download 48 GB Gemma weights). Rebuild all 30 Gemma layers with `constexpr_lut_to_dense` INT4pal (per grouped channel). Re-run ane-validator on L0, then golden-validator full-stack. If cosine ≥ 0.97 across all 30 layers and the 8-token golden prompt passes, this is T4.4 — the first fully-passing Gemma ANE full-stack. This is Exp 36.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="107-exp-35-complete-zaya1-8b-int4pal-t-1-win-speculative-decode-loss.html">Previous: Journal 107</a> | <a href="109-t4-3-closed-all-fp16-ane-inference-passes-golden-gate.html">Next: Journal 109</a></nav>
