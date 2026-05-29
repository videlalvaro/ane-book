---
layout: default
title: "Journal 088 - Structured CoT Grammar Decoding Investigation"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="087-phi-n-gram-speculative-upper-bound-simulated.html">Previous: Journal 087</a> | <a href="089-structured-cot-phi-ane-applicability-decision.html">Next: Journal 089</a></nav>

# 2026-04-29 - Structured CoT Grammar Decoding Investigation

**Intent**: Investigate Kaya Omer's Structured CoT / grammar-constrained scratchpad post as a possible Phi-4-mini ANE optimization, using Sakarovitch weighted automata and Dragon Book syntax-analysis framing to classify the technique before implementation.

**Setup**: Desk review only; target would be Phi-4-mini decode with CoreML/ANE layer and LM-head compute unchanged, plus a host-side FSM or grammar mask applied during sampling.

**Result**: Initial conclusion: this is guided decoding at sampling time, not a model-architecture change. It can fit the ANE-only mandate because matmuls, norms, attention, FFN, and LM-head projection remain in CoreML/ANE while CPU work only restricts token selection. Expected energy benefit is fewer generated tokens, not higher tok/s.

**Surprise / hurdle**: Phi-4-mini is not necessarily a reasoning model with native `<think>` behavior; exact grammars need tokenizer-aware literal sequences, and usefulness must be quality-gated on coding tasks rather than assumed from format compliance.

**Lesson**: Grammar-constrained scratchpads are a host policy for spending fewer decode steps, not an ANE throughput optimization.

**Next**: If pursued, prototype only the tokenizer-aware FSM/token-mask path and evaluate coding-task quality plus generated-token count before claiming energy gains.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="087-phi-n-gram-speculative-upper-bound-simulated.html">Previous: Journal 087</a> | <a href="089-structured-cot-phi-ane-applicability-decision.html">Next: Journal 089</a></nav>
