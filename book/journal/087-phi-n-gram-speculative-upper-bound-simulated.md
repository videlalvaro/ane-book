---
layout: default
title: "Journal 087 - Phi N-Gram Speculative Upper Bound Simulated"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="086-phi-code-shaped-n-gram-suite-measured.html">Previous: Journal 086</a> | <a href="088-structured-cot-grammar-decoding-investigation.html">Next: Journal 088</a></nav>

# 2026-04-29 - Phi N-Gram Speculative Upper Bound Simulated

**Intent**: Convert n-gram acceptance counts into an upper-bound target for a future public verifier, without claiming a runtime speedup yet.

**Setup**: Added helper script, which replays `Prompt IDs` and `Generated IDs` from the runtime log and simulates prompt-lookup draft blocks with configurable `--max-draft`.

**Result**: On the 5-prompt code-shaped suite, draft length 4 reduced ideal target verifier passes from 100 generated tokens to 49 verifier passes (`2.04x` pass-count upper bound). Draft length 8 reduced this to 41 verifier passes (`2.44x` upper bound). Both simulations used the same 69 accepted prompt-lookup tokens from the exact greedy log.

**Surprise / hurdle**: Larger draft length improves pass-count potential but also proposes farther beyond the first mismatch, so proposal acceptance rate over all proposed draft tokens falls. The right draft length will be a latency/acceptance tradeoff after a real verifier exists.

**Lesson**: The public algorithmic path is promising enough to justify a batch-token verifier artifact. The next bottleneck is not proposal quality; it is exact ANE-resident verification and KV commit/rollback.

**Next**: Design the smallest batch-token layer verifier around `T=4`, because it aligns with the already-gated batch-4 LM head and has a `~2x` pass-count target on code-shaped prompts.

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="086-phi-code-shaped-n-gram-suite-measured.html">Previous: Journal 086</a> | <a href="088-structured-cot-grammar-decoding-investigation.html">Next: Journal 088</a></nav>
