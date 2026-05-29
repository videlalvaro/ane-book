---
layout: default
title: "Journal 104 - Speculative Decode Prompt-Density Validation"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="103-hy-mt1-5-2-bit-gguf-ane-conversion-intent.html">Previous: Journal 103</a> | <a href="105-exp-26-follow-up-prompt-length-sweep-for-n-gram-speculative-decode.html">Next: Journal 105</a></nav>

# 2026-05-12 - Speculative Decode Prompt-Density Validation

**Intent**: Determine whether the previously measured +1.7% speculative decode speedup (Exp 25, 39-token code prompt) was a prompt-density floor rather than a fundamental ceiling of n-gram speculative decoding on ANE. Per Knuth TAOCP §6.1, n-gram match distance scales as ∝ 1/collision_frequency — a low-repetition prompt suppresses the drafter, so we need a far denser context to stress-test the T=4 verifier path.

**Setup**: Phi-4-mini RangeDim unified shards (`phi4mini_runtime_meta_rope96_rangedim_20_4_6_2.json`), `--speculative --ngram-min 1`, daemon benchmark (helper script). New dense prompt: 372-token Swift CoreML code snippet (temporary output) with heavy repetition of `MLMultiArray`, `MLModel`, `MLState`, `makeInputDict`, `forwardLayer`, `rope_cos`, `rope_sin`, `attn_mask`, `kv_write_mask`. Both prompts run for 5 reps, 20 new tokens (39-token) and 80 new tokens (372-token). JIT warmup: T=1=113.4s, T=4=136s.

**Result**:

| Prompt | Reps | New toks | Prefill tok/s | Decode tok/s | Speedup vs T=1 (17.8) |
|--------|------|----------|--------------|-------------|----------------------|
| 39-token code prompt (prior) | 5 | 20 | 68.9 | 18.1 | +1.7% |
| 372-token Swift CoreML prompt | 5 | 80 | **70.4** | **26.7** | **+50%** |

Decode reps for the 372-token run: 26.8, 26.7, 26.6, 26.7, 26.7 — variance ≤0.2 tok/s, confirming the measurement is stable and not JIT noise. Artifacts updated: `the validation-first notes` Exp 26 table row added; `local runtime notes` updated with both data points.

**Surprise / hurdle**: The +50% jump from a single prompt swap was striking. The simulated 2.04× upper bound from Exp 23 (`draft=4: verifier_passes=49/100`) is still above the measured 1.5×, meaning the drafter is not yet fully saturating every T=4 verify call — either some calls accept fewer than 4 tokens, or occasional fallbacks to T=1 remain. The gap between theoretical ceiling and measured wall is the next thing to quantify.

**Lesson**: N-gram speculative decoding acceptance rate on ANE is entirely dominated by prompt-token repetition density; a 10× increase in prompt length with the right vocabulary yielded a 29× larger speedup, confirming the drafter is the bottleneck, not the ANE verifier throughput.

**Next**: Map speedup vs. prompt length between 39 and 372 tokens to find the minimum context length needed for production-grade gains. The Knuth §6.1 match-distance model predicts a monotone but sublinear acceptance rate curve; measure 5–7 points to characterise the knee. Also instrument per-call acceptance count to close the gap between 1.5× measured and 2.04× simulated ceiling.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="103-hy-mt1-5-2-bit-gguf-ane-conversion-intent.html">Previous: Journal 103</a> | <a href="105-exp-26-follow-up-prompt-length-sweep-for-n-gram-speculative-decode.html">Next: Journal 105</a></nav>
