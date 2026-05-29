---
layout: default
title: "Journal 031 - Phi-4-mini Three-Layer Full-Shard Probe Passed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="030-phi-4-mini-three-layer-full-shard-probe-intent.html">Previous: Journal 030</a> | <a href="032-phi-4-mini-full-3-layer-shard-strategy-validation-intent.html">Next: Journal 032</a></nav>

# 2026-04-27 - Phi-4-mini Three-Layer Full-Shard Probe Passed

**Intent**: Try the user-requested 3-layer Phi-4-mini version to reduce layer CoreML calls from 32 to about 11, applying the validation-first notes call-hoisting/strength-reduction and whole-operation fusion discipline.

**Setup**: Built a non-destructive full INT8 stateful probe for layers 0–3 in local artifacts with `gguf_to_ane.py --layer-start 0 --layer-end 3 --output-name phi4mini_layer0_3_q8`; then compiled, ran strict MLComputePlan residency, and ran range golden quality. No perf/energy and no cleanup/deletion.

**Result**: PASS. Conversion and compile succeeded. Artifact sizes: `.mlpackage` 288M and `.mlmodelc` 288M. Strict residency passed: conv_total=12 conv_ane=12 conv_non_ane=0; compute_total=438 compute_ane=438 compute_non_ane=0. Quality range smoke passed: cos_hidden=0.999768, rmse=0.013637, max_abs=0.094727.

**Surprise / hurdle**: The 288M compiled artifact exceeded the older conservative ~250 MB shard-size caution line but still compiled and remained fully ANE-resident for this range.

**Lesson**: Three-layer Phi-4-mini fusion is promising for reducing call count, but shard-size guidance must be revalidated per layer range rather than assumed from one successful 288M compile.

**Next**: Validate 3-layer shards across all layer ranges before any scale-out/perf/energy claim; keep cleanup/deletion out of this probe path.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="030-phi-4-mini-three-layer-full-shard-probe-intent.html">Previous: Journal 030</a> | <a href="032-phi-4-mini-full-3-layer-shard-strategy-validation-intent.html">Next: Journal 032</a></nav>
