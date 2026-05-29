---
layout: default
title: "Journal 102 - Phi-4-mini Rope96 Fast Fused Rebuild Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="101-phi-4-mini-rope96-fast-fused-rebuild-intent.html">Previous: Journal 101</a> | <a href="103-hy-mt1-5-2-bit-gguf-ane-conversion-intent.html">Next: Journal 103</a></nav>

# 2026-04-30 - Phi-4-mini Rope96 Fast Fused Rebuild Outcome

**Intent**: Record completion of the Phi-4-mini Rope96 fast fused rebuild so the fastest public topology is available with the corrected partial-RoPE contract. This follows validation-before-performance discipline plus Dragon Book call-hoisting/strength-reduction and Iverson whole-operation fusion: reduce CoreML layer-call count only after the rebuilt shards re-pass ANE residency and range golden.

**Setup**: Rebuilt the fixed `rope_dim=96` INT8 fused topology [0,20)+[20,24)+[24,30)+[30,32). Runtime manifest: local artifacts. Swift runtime rebuilt as local artifacts, and defaults now point to the fast manifest. The CLI now prints startup, prefill, and decode timing and uses one warmup call by default. `.mlpackage` intermediates were removed after the compiled runtime artifacts were gated.

**Result**: PASS. All fused shards passed strict ANE residency and range golden: [0,20) cos=0.9985457359614087 with compute_ane=2983/2983; [20,24) cos=0.9994461151166388 with compute_ane=599/599; [24,30) cos=0.999453852407371 with compute_ane=897/897; [30,32) cos=0.9997610015303526 with compute_ane=301/301. The Erlang hello-world smoke is correct. Warm smoke measured prefill_tok_s=16.113, decode_tok_s=16.602, request_s=3.111. A cold no-warmup first request showed about 115s prefill from lazy CoreML activation.

**Surprise / hurdle**: The rebuilt fast topology stayed fully ANE-resident with high range-golden cosine, but cold no-warmup timing is dominated by lazy CoreML activation and can dwarf the actual warmed request latency.

**Lesson**: The corrected Rope96 fast fused Phi topology is usable only when reported with explicit startup/warmup timing; warm decode speed and cold activation cost are different phenomena.

**Next**: Treat `phi4mini_runtime_meta_rope96_fast_20_4_6_2.json` and `phi4_mini_ane_runtime_rope96` as the current Rope96 fast public runtime baseline; future comparisons should use the default one-call warmup, preserve ANE residency and range-golden gates, and separate cold-start activation from warmed prefill/decode throughput.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md); [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="101-phi-4-mini-rope96-fast-fused-rebuild-intent.html">Previous: Journal 101</a> | <a href="103-hy-mt1-5-2-bit-gguf-ane-conversion-intent.html">Next: Journal 103</a></nav>
