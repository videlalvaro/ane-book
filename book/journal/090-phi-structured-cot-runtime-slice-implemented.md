---
layout: default
title: "Journal 090 - Phi Structured CoT Runtime Slice Implemented"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="089-structured-cot-phi-ane-applicability-decision.html">Previous: Journal 089</a> | <a href="091-phi-n-gram-force-mode-tried.html">Next: Journal 091</a></nav>

# 2026-04-29 - Phi Structured CoT Runtime Slice Implemented

**Intent**: Ship a minimal public Phi-4-mini structured-decoding feature quickly, without touching model artifacts or the default greedy path.

**Setup**: Added helper script, generated local artifacts, and wired `--structured-cot` / `--structured-cot-manifest` into [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift).

**Result**: Runtime now supports a tokenizer-aware FSM with literal, field, and open stages. Literal stages force exact token IDs and skip LM-head prediction while still running the ANE layer stack for KV correctness. Field stages use constrained argmax and block stop tokens until newline is allowed or forced by budget. JSONL serve can also request `structured_cot` when a manifest is loaded.

**Validation**: `swiftc -O -c [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) -o temporary output -framework CoreML -framework Foundation` passed; `.venv/bin/python -m py_compile helper script` passed. Built local artifacts and ran `--meta local artifacts --max-new 16 --structured-cot --profile` successfully.

**Smoke numbers**: `decode_tok_s=16.609`, `layers_ms=56.151`, `head_predict_reduce_ms=4.049`, `forced_tokens=6`, `field_content_tokens=10`, `fields_completed=0` in the short budget.

**Lesson**: The hook is now shippable as an opt-in runtime policy. It proves the host FSM can constrain Phi output while preserving the ANE compute path. It does not yet prove coding quality or energy improvement.

**Next**: Run a longer coding-prompt suite that reaches `CODE:` and compare unconstrained vs structured mode on total tokens, code extraction, syntax/pass proxies, and energy per solved task.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift); [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="089-structured-cot-phi-ane-applicability-decision.html">Previous: Journal 089</a> | <a href="091-phi-n-gram-force-mode-tried.html">Next: Journal 091</a></nav>
