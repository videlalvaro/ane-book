---
layout: default
title: "Journal 086 - Phi Code-Shaped N-Gram Suite Measured"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="085-phi-n-gram-proposal-probe-added.html">Previous: Journal 085</a> | <a href="087-phi-n-gram-speculative-upper-bound-simulated.html">Next: Journal 087</a></nav>

# 2026-04-29 - Phi Code-Shaped N-Gram Suite Measured

**Intent**: Measure n-gram/prompt-lookup acceptance on more coding-like token histories, not just the degenerate repetitive smoke prompt.

**Setup**: Added `--prompt-ids-file` support to [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift) so multiple prompts can reuse one loaded runtime. Added helper script to regenerate a small code-shaped prompt-ID suite from the local Phi GGUF tokenizer metadata. Ran 5 prompts with `--max-new 24 --ngram-probe --ngram-min 2 --ngram-max 8` on the `20+4+6+2` manifest.

**Result**: Aggregate suite result: `targets=100`, `proposals=74`, `accepted=69`, `proposal_rate=0.740`, `acceptance_rate=0.932`, `accepted_per_target=0.690`. By n-gram size: `N2=5/7`, `N3=4/5`, `N4=5/5`, `N5=5/5`, `N6=5/5`, `N7=4/5`, `N8=41/42`.

**Surprise / hurdle**: Acceptance is high enough to be interesting, but this still does not reduce latency until a verifier can validate and commit multiple proposed tokens while keeping KV state correct.

**Lesson**: Prompt-lookup speculation has real signal on code-shaped token streams. The next public optimization should target verifier mechanics, not private API roundtrips.

**Next**: Build the smallest public batch-token verifier probe or prove that public `MLState` cannot support cheap rollback/commit; keep baseline greedy decode unchanged until exactness is preserved.

**Refs**: [runtime/phi4_mini_ane.swift](https://github.com/videlalvaro/ane-models/blob/main/runtime/phi4_mini_ane.swift)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="085-phi-n-gram-proposal-probe-added.html">Previous: Journal 085</a> | <a href="087-phi-n-gram-speculative-upper-bound-simulated.html">Next: Journal 087</a></nav>
