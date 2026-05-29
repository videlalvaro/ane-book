---
layout: default
title: "Journal 105 - Exp 26 Follow-Up: Prompt-Length Sweep for N-Gram Speculative Decode"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="104-speculative-decode-prompt-density-validation.html">Previous: Journal 104</a> | <a href="106-exp-35-zaya1-8b-moe-int4-per-grouped-channel-palettization-intent.html">Next: Journal 106</a></nav>

# 2026-05-12 - Exp 26 Follow-Up: Prompt-Length Sweep for N-Gram Speculative Decode

**Intent**: Characterise how n-gram speculative decode speedup scales with context length by running a 4-point sweep (100 → 200 → 372 → 800 tokens), working toward the 2.04× simulated upper bound established in Exp 23. Per Knuth TAOCP §6.1, match-collision frequency grows with context density, predicting a monotone but sublinear acceptance-rate curve; the sweep is the empirical trace of that curve.

**Setup**: Runtime local artifacts; shards local artifacts (RangeDim T=1..4, 100% ANE residency, topology 20+4+6+2); manifest `phi4mini_runtime_meta_rope96_rangedim_20_4_6_2.json`. Prompts tiled from temporary output (dense Swift CoreML code: `MLMultiArray`, `MLModel`, etc.). 5 reps per length, 80 new tokens per request. Single daemon session; JIT paid once (T=1 JIT 113.4s, T=4 JIT 140.8s). Sweep script: helper script. Raw log: temporary output.

**Result**:

| Prompt length | Decode tok/s | Prefill tok/s | Wall/req | Speedup vs T=1 (17.8) |
|--------------|-------------|--------------|---------|----------------------|
| 100 tokens   | 21.1        | 70.1         | 5.17s   | 1.19×                |
| 200 tokens   | 22.1        | 70.3         | 6.42s   | 1.24×                |
| 372 tokens   | 26.7        | 70.1         | 8.26s   | 1.50×                |
| 800 tokens   | 28.9        | 69.9         | 14.19s  | 1.62×                |

Prefill stable at ~70 tok/s across all lengths (T=4 chunked path scales cleanly). Decode speedup is monotonically rising and not yet saturated at 800 tokens. Artifacts: helper script (sweep script), temporary output (raw output), `the validation-first notes` Exp 26 section updated with prompt-length sweep table, `local runtime notes` updated with sweep curve data.

**Surprise / hurdle**: The 1.62× at 800 tokens approaches but has not reached the 2.04× simulated ceiling, meaning the acceptance rate is still climbing. The gap implies either some T=4 verify calls accept fewer than 4 tokens or occasional fallbacks to T=1 persist at longer contexts. The sweep also reveals that the 372-token point is squarely mid-curve, not near saturation — previous Exp 26 reports should not be cited as a plateau.

**Lesson**: N-gram acceptance rate is strongly context-density-dependent and has not saturated by 800 tokens; any speedup claim should always state the prompt length alongside it.

**Next**: Extend the sweep to 1200–2048 tokens to find the saturation knee; instrument per-call draft acceptance count to close the measured-vs-simulated ceiling gap; if the curve has not flattened by 2048 tokens, revisit the Exp 23 upper-bound simulation assumptions.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="104-speculative-decode-prompt-density-validation.html">Previous: Journal 104</a> | <a href="106-exp-35-zaya1-8b-moe-int4-per-grouped-channel-palettization-intent.html">Next: Journal 106</a></nav>
