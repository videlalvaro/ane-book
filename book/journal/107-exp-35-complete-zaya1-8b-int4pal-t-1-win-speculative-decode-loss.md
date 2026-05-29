---
layout: default
title: "Journal 107 - Exp 35 COMPLETE: ZAYA1-8B INT4pal T=1 Win, Speculative Decode Loss"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="106-exp-35-zaya1-8b-moe-int4-per-grouped-channel-palettization-intent.html">Previous: Journal 106</a> | <a href="108-exp-36-intent-gemma-4-t4-3-root-cause-identified-rebuild-to-int4pal.html">Next: Journal 108</a></nav>

# 2026-05-13 - Exp 35 COMPLETE: ZAYA1-8B INT4pal T=1 Win, Speculative Decode Loss

**Intent**: Validate INT4 palettization (`constexpr_lut_to_dense`, `group_size=32`) on ZAYA1-8B MoE FFN shards as a bandwidth-reduction upgrade from the INT8 baseline, then test whether speculative decode (T=4 MoE verifier + T=1 draft) converts the halved shard size into throughput gains. Full intent logged in prior session (Exp 35 intent entry above). Citations: EoP §4 (semigroup element size reduction); TAOCP Vol. 2 §4.3 (arithmetic vs. memory bottleneck identification).

**Setup**: ZAYA1-8B MoE (40 alternating MoE layers). INT4pal shards built to `<external-scratch>/zaya_shards/` (40 × `.mlmodelc`). T=1 baseline timing: wall-clock tok/s measured on warm-cache decode runs. Speculative decode setup: T=4 verifier (processes 4 candidate tokens per call) using full MoE shards; T=1 draft using same model. Acceptance rate measured on synthetic prompts and code-completion prompts. Break-even formula: `p_break_even = 1 − t1 / (tv / vbt)` where `t1 = 109 ms` (T=1 step), `tv = 483 ms` (T=4 verifier call), `vbt = 4` (verifier batch size).

**Result**: INT4pal shards: 40 `.mlmodelc` files at 101.2 MB each (vs. ~202 MB INT8 baseline — 50% reduction as predicted). ANE residency: 100% confirmed. T=1 throughput: **9.25 tok/s** (+7.7% over INT8 8.59 tok/s baseline). Speculative decode (T=4 verifier + T=1 draft): **2.52 tok/s** — significantly slower than baseline. Measured acceptance rate on synthetic prompts: 7.3%. Calculated break-even acceptance rate: ~10%. Code-completion prompts (60–80% acceptance) still yield only ~7.3 tok/s effective — slower than 9.25 tok/s T=1 baseline because the 483 ms verifier dominates.

**Surprise / hurdle**: The halved shard size did improve T=1 throughput exactly as bandwidth-bound theory predicts. However, the T=4 verifier is MAC-bound (not memory-bandwidth-bound) at 483 ms — halving weight size did not halve verifier latency. The break-even math (p_accept ≥ 0.10) was only barely plausible on paper; real synthetic prompt acceptance at 7.3% is below it. Even high-quality code prompts at 60–80% acceptance fail to overcome the 483 ms fixed verifier cost. The soft routing in ZAYA (dense MoE, not top-K sparse) forces all experts, making verifier cost disproportionate relative to the draft cost.

**Lesson**: INT4pal is a bandwidth win for T=1 on memory-bound MoE inference; it does NOT reduce the MAC cost of a multi-token verifier — speculative decode on dense MoE requires either sparse top-K routing (fewer activated experts → lower verifier MAC) or an architecture where attention dominates compute (draft and verifier have similar cost).

**Next**: INT4pal T=1 at 9.25 tok/s is the new ZAYA production baseline. Speculative decode on ZAYA requires sparse top-K routing or a different draft architecture — not pursued further on this model. Pivot to Gemma 4 T4.3 full-stack quality failure (Exp 36).

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="106-exp-35-zaya1-8b-moe-int4-per-grouped-channel-palettization-intent.html">Previous: Journal 106</a> | <a href="108-exp-36-intent-gemma-4-t4-3-root-cause-identified-rebuild-to-int4pal.html">Next: Journal 108</a></nav>
