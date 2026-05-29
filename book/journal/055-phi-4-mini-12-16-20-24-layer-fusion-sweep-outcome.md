---
layout: default
title: "Journal 055 - Phi-4-mini 12/16/20/24-Layer Fusion Sweep Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="054-phi-4-mini-twenty-layer-front-shard-intent.html">Previous: Journal 054</a> | <a href="056-private-ane-chaining-investigation-intent.html">Next: Journal 056</a></nav>

# 2026-04-28 - Phi-4-mini 12/16/20/24-Layer Fusion Sweep Outcome

**Intent**: Complete the larger front-fused Phi-4-mini sweep to locate the strict ANE residency cliff and throughput sweet spot after 8-layer asymmetric fusion, applying the validation-first notes Iverson/APL whole-array fusion and Dragon Book call-hoisting while preserving residency and golden gates.

**Setup**: Tested larger INT8 stateful front shards and asymmetric manifests using already validated tail shards: 12+12+6+2, 16+8+6+2, 20+4+6+2, and the attempted 24+6+2 path. Manifests profiled included `phi4mini_runtime_meta_12_12_6_2.json`, `phi4mini_runtime_meta_16_8_6_2.json`, and `phi4mini_runtime_meta_20_4_6_2.json`. No cleanup/deletion or energy benchmark was performed.

**Result**: [0,12) passed strict residency with 48/48 conv on ANE and golden cos=0.999197, rmse=0.039475, max_abs=0.228516. [12,24) passed strict residency and golden cos=0.997967, rmse=0.166971, max_abs=1.71875. The 12+12+6+2 manifest `phi4mini_runtime_meta_12_12_6_2.json` profiled short at 16.598 then 17.159 tok/s, and long max-new=64 at 16.659 tok/s with layers=54.865 ms/token. [0,16) compiled at 1.5G, passed strict residency with 64/64 conv on ANE, and golden cos=0.998717, rmse=0.057385, max_abs=0.308594. The 16+8+6+2 manifest `phi4mini_runtime_meta_16_8_6_2.json` profiled short at 16.669 then 17.174 tok/s, and long max-new=64 at 17.143 tok/s with layers=53.225 ms/token; this is the current best. [0,20) compiled at 1.9G, passed strict residency with 80/80 conv on ANE, and golden cos=0.998546, rmse=0.096738, max_abs=0.421875. The 20+4+6+2 manifest `phi4mini_runtime_meta_20_4_6_2.json` profiled around 16.65-16.70 tok/s, and long max-new=64 at 16.697 tok/s with layers=54.742 ms/token. [0,24) compiled at 2.3G but failed strict residency completely: conv_total=96, conv_ane=0, compute_ane=0, all CPU, so it was disqualified before golden validation.

**Surprise / hurdle**: Compile success scaled to a 2.3G [0,24) artifact, but CoreML placed the entire graph on CPU, making strict residency rather than compile size the hard acceptance boundary.

**Lesson**: For this Phi-4-mini graph/compiler, the ANE residency cliff is between 20 and 24 fused front layers, and the best measured performance topology is 16+8+6+2.

**Next**: Treat 16+8+6+2 as the current runtime baseline for future comparisons; do not use [0,24) or any CPU-placed fused shard, and keep subsequent optimization behind strict residency plus golden validation.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="054-phi-4-mini-twenty-layer-front-shard-intent.html">Previous: Journal 054</a> | <a href="056-private-ane-chaining-investigation-intent.html">Next: Journal 056</a></nav>
