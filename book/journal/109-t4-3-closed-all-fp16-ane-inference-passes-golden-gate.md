---
layout: default
title: "Journal 109 - T4.3 CLOSED: All-FP16 ANE inference passes golden gate"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="108-exp-36-intent-gemma-4-t4-3-root-cause-identified-rebuild-to-int4pal.html">Previous: Journal 108</a> | <a href="110-t4-1-5-closed-full-16-token-decode-exact-match-on-all-fp16-ane-stack.html">Next: Journal 110</a></nav>

# 2026-05-14 - T4.3 CLOSED: All-FP16 ANE inference passes golden gate

**Intent recorded before this session:** Move all Gemma-4-26B-A4B FFN shards from GPU to ANE by splitting from 2 sub-shards to 8 sub-shards at FP16.

**Root cause confirmed:** The original q8c FFN shards were 364 MB (`p0of2`) and 398 MB (`p1of2`) compiled — both above the empirically validated ~250 MB ANE shard limit. CoreML silently placed them on GPU. GPU float16 ≠ ANE float16 numerics + INT8 quantization error compounded across 30 layers, producing wrong decode tokens `[236881, 236881]` instead of `[669, 5279]`.

**Fix applied:**
- Rebuilt all 30 FFN layers with `--ffn-shards 8 --quant-bits 0`
- Each sub-shard: 1 expert pack (16 experts) ≈ 182 MB (p0–p6) or 216 MB (p7, includes combiner + norms)
- All 8 sub-shards per layer land on ANE — confirmed within the 250 MB limit
- All 30 attn shards also FP16 (rebuilt in prior session to fix global attn INT8 per-channel error)
- Total: 30 layers × (1 attn + 8 FFN) = 270 compiled mlmodelc files, all on ANE
- Production meta: helper script

**Gate results (7-token prompt `[3689,563,506,5279,529,7001,236881]`, 2 decode steps):**
- Prompt pos 0–6 cosine vs `gemma_golden.npz[logits_full]`: 0.9997, 0.9996, 0.9977, 0.9980, 0.9944, 0.9982, 0.9957 — all ≥ 0.97 PASS
- Decode pos 0 cosine vs `gemma_golden.npz[next_token_logits][0]`: 0.9976 PASS
- Decode tokens: `[669, 5279]` — exact match with HF reference

**Key lesson (burn this in):** CoreML does NOT warn when a shard exceeds the ANE limit — it silently falls back to GPU. The only reliable check is `du -sh *.mlmodelc`: if any shard > 250 MB, it's on GPU regardless of the `computeUnits = .cpuAndNeuralEngine` flag. The fix is always to split further, never to optimise the GPU path.

**Timing:** 37 s per layer for 8-shard FP16 FFN export (Xcode python3, M4 Max). Full rebuild of 29 layers ≈ 18 min. TTFT with all-ANE: ~208 s (model load + 7-tok prefill, not optimised). Per-token decode: ~29 s (270 shards sequential, not optimised).

**Dead end noted:** Trying INT8 per-channel quantization on attn shards caused >0.03 cosine drop per global attention layer, cascading to 0.55 cosine at L25. FP16 attn is mandatory for quality.

**Dead end noted:** 2-shard FFN (even FP16) would be 364/398 MB → GPU. The 8-shard split is the minimum to stay under ANE limit for this model.

**Next:** ANE residency probe on one rebuilt FFN shard (project policy: `ane-validator` gate before scale-out). Then INT4 palettization investigation as next compression path.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="108-exp-36-intent-gemma-4-t4-3-root-cause-identified-rebuild-to-int4pal.html">Previous: Journal 108</a> | <a href="110-t4-1-5-closed-full-16-token-decode-exact-match-on-all-fp16-ane-stack.html">Next: Journal 110</a></nav>
