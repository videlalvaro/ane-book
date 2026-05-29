---
layout: default
title: "Journal 047 - Phi-4-mini Five- and Six-Layer Fusion Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="046-phi-4-mini-six-layer-fused-strategy-intent.html">Previous: Journal 046</a> | <a href="048-phi-4-mini-lm-head-optimization-outcome.html">Next: Journal 048</a></nav>

# 2026-04-28 - Phi-4-mini Five- and Six-Layer Fusion Outcome

**Intent**: Push Phi-4-mini decode tok/s higher after the 4-layer fused topology reached 15.412 tok/s in the same session, using the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Built, validated, and profiled full 5-layer topology ranges [0,5), [5,10), [10,15), [15,20), [20,25), [25,30), [30,32), then full 6-layer topology ranges [0,6), [6,12), [12,18), [18,24), [24,30), [30,32). 5-layer artifacts: six 481M shards plus 192M tail; runtime manifest local artifacts. 6-layer artifacts: five 577M shards plus 192M tail; runtime manifest local artifacts.

**Result**: 5-layer residency passed all ranges: fused shards conv=20/20, compute=728/728; tail conv=8/8, compute=293/293; zero non-ANE. 5-layer golden passed all ranges: cos min/mean/max=0.999227/0.999458/0.999761, rmse max=0.167865, max_abs max=0.721680. 5-layer best repeat profile: decode_tok_s=15.661, layers_ms=58.754, head_predict_reduce_ms=5.093, layer_shards=7. 6-layer residency passed all ranges: fused shards conv=24/24, compute=873/873; tail conv=8/8, compute=293/293; zero non-ANE. 6-layer golden passed all ranges: cos min/mean/max=0.999072/0.999362/0.999761, rmse max=0.422014, max_abs max=5.437500. 6-layer profile: best decode_tok_s=16.103, repeat decode_tok_s=15.726, layers_ms best=57.001, head_predict_reduce_ms about 5.09-5.12, layer_shards=6.

**Surprise / hurdle**: The 6-layer topology produced the first >16 tok/s run, but fusion gains are diminishing and late-layer [24,30) shows larger absolute drift despite high cosine.

**Lesson**: Deeper ANE layer fusion can still buy decode throughput, but the remaining host-side head path is now a larger lever than further shard fusion.

**Next**: Move the LM-head top-k/argmax path onto ANE to reduce the remaining ~5.1 ms/token head path; no deletion/cleanup and no energy benchmark were run.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="046-phi-4-mini-six-layer-fused-strategy-intent.html">Previous: Journal 046</a> | <a href="048-phi-4-mini-lm-head-optimization-outcome.html">Next: Journal 048</a></nav>
