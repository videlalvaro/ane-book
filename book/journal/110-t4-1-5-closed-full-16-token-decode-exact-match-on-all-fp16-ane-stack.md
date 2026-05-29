---
layout: default
title: "Journal 110 - T4.1.5 CLOSED: Full 16-token decode exact match on all-FP16 ANE stack"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="109-t4-3-closed-all-fp16-ane-inference-passes-golden-gate.html">Previous: Journal 109</a> | <a href="111-o2-concurrent-ffn-partial-fan-out.html">Next: Journal 111</a></nav>

# 2026-05-14 - T4.1.5 CLOSED: Full 16-token decode exact match on all-FP16 ANE stack

**Intent**: Verify that the all-FP16 ANE stack (T4.3, all 270 shards on ANE) produces correct output across a full 16-token autoregressive decode, closing the T4 correctness milestone.

**Setup**: Runtime: `gemma_swift_head_meta_allfp16.json`, 270 shards (30 attn + 30×8 FFN, all FP16, all ANE). Prompt: `[3689, 563, 506, 5279, 529, 7001, 236881]` (7 tokens). Decode: `--n-new 16`. Reference: `gemma_golden.npz[next_token_ids]`. Hardware: M4 Max, no sudo, unoptimised sequential shard-reload path.

**Result**: Generated `[669, 5279, 529, 7001, 236881, 669, 5279, 529, 7001, 236881, 669, 5279, 529, 7001, 236881, 669]` — exact 16/16 match. T4 correctness milestone closed. Timing baseline: TTFT ~212 s (model load + 7-tok prefill), decode 28.9 s/tok (0.034 tok/s), 270 shards sequential per token.

**Surprise / hurdle**: The prior investigation (2026-04-24, Row 7 divergence `506` → `9405`) required rounds of hidden-boundary attribution, gamma amplification analysis, and layer-27/28/29 debug taps before the GPU-FFN root cause was confirmed. In hindsight, `du -sh *.mlmodelc` would have identified the over-limit shards in seconds. The debugging effort was several days; the fix was one build flag (`--ffn-shards 8`).

**Lesson**: Before debugging hidden-state divergence in a multi-shard ANE stack, always check `du -sh *.mlmodelc` first — any shard > 250 MB is silently on GPU, and GPU numerical drift compounding across 30 layers is indistinguishable from a model bug without this check.

**Next**: T4 correctness is closed. The ANE chain primitive work (Rounds 2–3 in ANE_CHAIN_SCHEMA.md) — eliminating per-token shard reload overhead — is now the primary performance path. The 28.9 s/tok figure is the unoptimised correctness baseline to beat.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="109-t4-3-closed-all-fp16-ane-inference-passes-golden-gate.html">Previous: Journal 109</a> | <a href="111-o2-concurrent-ffn-partial-fan-out.html">Next: Journal 111</a></nav>
