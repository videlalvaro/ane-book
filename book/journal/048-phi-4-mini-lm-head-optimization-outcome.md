---
layout: default
title: "Journal 048 - Phi-4-mini LM-Head Optimization Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="047-phi-4-mini-five-and-six-layer-fusion-outcome.html">Previous: Journal 047</a> | <a href="049-phi-4-mini-eight-layer-fused-shard-intent.html">Next: Journal 049</a></nav>

# 2026-04-28 - Phi-4-mini LM-Head Optimization Outcome

**Intent**: Reduce the remaining Phi-4-mini LM-head bottleneck after 6-layer fusion made the layer chain faster, following measurement-before-optimization discipline and the project ANE-only mandate for compute-heavy projection/reduction work.

**Setup**: Tested ANE-resident LM-head alternatives on the 6-layer fused runtime: an experimental top-1 LM-head shard, an 8-way full-logit LM head under local artifacts, and a 3-way full-logit LM head under local artifacts. Runtime manifests included local artifacts and local artifacts. Swift runtime now supports variable LM-head shard counts and profiling counters that separate head predict shard work from host reduce work.

**Result**: The experimental top-1 LM-head shard compiled but failed strict residency because `ios18.topk` and `cast` landed on CPU. The 8-way full-logit LM head built successfully; all shards were ANE-resident and golden-passed, but runtime did not improve: about 5.223 ms/token head versus the 4-way baseline at about 5.156 ms/token. The 3-way full-logit LM head also built successfully; all shards were ANE-resident and golden-passed, with about 5.13 ms/token head, essentially tied with 4-way. Profiling showed the host local argmax scan costs only about 0.25-0.27 ms/token, so the LM-head bottleneck is CoreML/ANE predict latency rather than Swift reduction.

**Surprise / hurdle**: `torch.topk` lowered through CoreML into CPU-side `ios18.topk`/`cast` for this pattern, while changing the number of full-logit shards shifted predict overhead only slightly and did not remove the about 5 ms/token head floor.

**Lesson**: The Phi-4-mini LM-head bottleneck is not the Swift argmax reduction; it is the CoreML/ANE predict cost of evaluating the full vocabulary projection shards.

**Next**: True ANE-resident reduction/top-k needs a different CoreML op pattern because `torch.topk` lowers to CPU here; otherwise the next likely avenues are reducing LM-head projection size or avoiding a full head on every token via vocabulary, routing, or speculative approaches, all behind residency and golden quality gates.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="047-phi-4-mini-five-and-six-layer-fusion-outcome.html">Previous: Journal 047</a> | <a href="049-phi-4-mini-eight-layer-fused-shard-intent.html">Next: Journal 049</a></nav>
