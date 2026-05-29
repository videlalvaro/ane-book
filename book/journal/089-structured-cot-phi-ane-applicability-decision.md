---
layout: default
title: "Journal 089 - Structured CoT Phi/ANE Applicability Decision"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="088-structured-cot-grammar-decoding-investigation.html">Previous: Journal 088</a> | <a href="090-phi-structured-cot-runtime-slice-implemented.html">Next: Journal 090</a></nav>

# 2026-04-29 - Structured CoT Phi/ANE Applicability Decision

**Intent**: Turn the Structured CoT investigation into a concrete yes/no implementation decision for the Phi-4-mini public ANE runtime.

**Setup**: Checked the current Swift sampling path and the local Phi GGUF tokenizer metadata. The runtime already performs ANE layer execution and ANE LM-head projection, then chooses the argmax on host. The tokenizer has newline and punctuation tokens, but structured literals such as `<think>` and `GOAL:` are not single tokens.

**Result**: Decision: applicable, but as constrained decoding and token-budget control, not per-token acceleration. A grammar/FSM can constrain host-side argmax to valid next tokens while keeping all heavy compute on ANE. Best first grammar should be Phi-specific and visible-plan oriented (`GOAL/STATE/ALGO/EDGE/VERIFY` or similar) rather than assuming Qwen-style native `<think>` behavior.

**Surprise / hurdle**: Forced grammar literals are not free. They still need ANE layer passes to update KV cache. The only immediate compute saving on forced literals would be skipping LM-head prediction via a future layer-only `advanceOnly` path, which saves the `~5 ms/token` head cost but not the `~53 ms/token` layer stack.

**Lesson**: Structured CoT is complementary to n-gram/speculation. Structured decoding can reduce total tokens and improve code extraction reliability; n-gram/speculation can reduce verifier passes if a batch verifier exists.

**Next**: Prototype a tokenizer-aware grammar manifest plus constrained argmax in `phi4_mini_ane.swift`; measure unconstrained vs grammar-constrained on the code-shaped prompt suite before building any larger harness.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-book/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="088-structured-cot-grammar-decoding-investigation.html">Previous: Journal 088</a> | <a href="090-phi-structured-cot-runtime-slice-implemented.html">Next: Journal 090</a></nav>
