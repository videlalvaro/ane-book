---
layout: default
title: "Journal 099 - Phi Full-Stack GGUF Reference Gate Blocks Q8 Chat"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="098-phi-4-mini-t-4-speculative-exactness-comparison-intent.html">Previous: Journal 098</a> | <a href="100-phi-4-mini-partial-rope-patch-and-probe-passed.html">Next: Journal 100</a></nav>

# 2026-04-29 - Phi Full-Stack GGUF Reference Gate Blocks Q8 Chat

**Intent**: Record the full-stack Phi-4-mini GGUF reference result as a validation gate only, not an inference shortcut, following validation-before-performance discipline and Knuth-style end-to-end sequential verification.

**Setup**: Added and ran helper script as a CPU/PyTorch GGUF fp16 reference for Phi-4-mini validation. Prompt: `<|system|>You are a helpful assistant. Answer briefly.<|end|><|user|>write hello world in Erlang<|end|><|assistant|>`. Prompt IDs: `[200022,3575,553,261,10297,29186,13,30985,51088,13,200020,200021,9566,40617,2375,306,101038,516,200020,200019]`.

**Result**: The GGUF fp16 reference predicted first token `168394`, decoded as a code fence token, with top8 `[168394,1,1385,95839,62915,26178,185334,13225]`. A 16-token reference run generated IDs `[168394,259,7585,198,1314,61400,8595,271,2123,568,13225,11,5922,0,200020]`, decoded roughly as a code-fence/code path. The current ANE q8 runtime previously predicted `182298` and went into a Russian greeting path. Existing per-layer, multi-token state, and LM-head shard gates were green.

**Surprise / hurdle**: Shard-local gates were insufficient: all local quality/residency checks can pass while full-stack chat prefill+decode still diverges at the first generated token on a real prompt.

**Lesson**: Phi q8 chat must remain blocked until a full-stack prefill+decode golden gate passes against the GGUF reference.

**Next**: Run an ANE-vs-reference layer trace on the real prompt to localize the drift; consider targeted FP16 or mixed rebuild only after localization, ANE residency, and full-stack quality gates pass. Do not run long conversions from this finding alone.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="098-phi-4-mini-t-4-speculative-exactness-comparison-intent.html">Previous: Journal 098</a> | <a href="100-phi-4-mini-partial-rope-patch-and-probe-passed.html">Next: Journal 100</a></nav>
