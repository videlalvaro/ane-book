---
layout: default
title: "Journal 082 - Phi Dead Artifact Cleanup Approval Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="081-phi-batch-4-lm-head-full-set-gated.html">Previous: Journal 081</a> | <a href="083-phi-dead-artifact-cleanup-outcome.html">Next: Journal 083</a></nav>

# 2026-04-29 - Phi Dead Artifact Cleanup Approval Intent

**Intent**: Record the user's explicit approval to delete strong dead Phi artifacts and reclaim disk while preserving small tracked manifests where possible; this follows measurement discipline by removing only paths already ruled out by residency or slower-profile evidence.

**Setup**: Approved targets are the rejected `phi4_mini_ane_24layer_probe` [0,24) CPU-fallback artifact; generated top-k LM-head artifacts that failed ANE residency because `ios18.topk`/`ios18.cast` lowered to CPU; generated 3-way and 8-way LM-head artifacts that profiled slower than the 4-way head; and known slower 5-layer tail artifacts [20,25), [25,30), plus duplicate [30,32).

**Result**: Cleanup approval logged; deletion itself is a separate destructive action and should be limited to the approved dead artifacts.

**Surprise / hurdle**: Disk pressure forced distinguishing failed/slower experiment artifacts from the current working baseline instead of doing broad cleanup.

**Lesson**: Artifact cleanup is safe only when each deletion target has a recorded rejection reason and current baselines are explicitly protected.

**Next**: Delete only the approved dead artifacts if cleanup proceeds; do not delete the current baseline artifacts or any batch-4 LM-head artifacts.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="081-phi-batch-4-lm-head-full-set-gated.html">Previous: Journal 081</a> | <a href="083-phi-dead-artifact-cleanup-outcome.html">Next: Journal 083</a></nav>
