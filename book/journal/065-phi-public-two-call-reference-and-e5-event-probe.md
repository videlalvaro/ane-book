---
layout: default
title: "Journal 065 - Phi Public Two-Call Reference and E5 Event Probe"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="064-phi-stateful-raw-e5rt-chain-smoke.html">Previous: Journal 064</a> | <a href="066-phi-e5-objc-sync-point-experiment.html">Next: Journal 066</a></nav>

# 2026-04-28 - Phi Public Two-Call Reference and E5 Event Probe

**Intent**: Prove whether the raw Phi stage-B zero was caused by bad synthetic inputs or by the private two-operation E5 stream path.

**Setup**: Added local artifacts, a public-only CoreML reference that loads `phi4mini_layer16_24_q8.mlmodelc` and `phi4mini_layer24_30_q8.mlmodelc` in a fresh process. It runs public stateful prediction for stage A, feeds A's hidden to stage B, and avoids mixing public prediction with private operation-pool mutation.

**Result**: Public stage A exactly matches raw stage A for the synthetic Phi input: sample `[1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]`, sum `-337.912079`. Public stage B is nonzero: sample `[-8.0625, -0.251953125, -0.564453125, -5.12890625, ...]`, sum `-166.729431`. The raw E5 path still returns all-zero stage B, so the remaining issue is not input generation; it is a raw multi-op dependency, event, or output synchronization problem.

**Surprise / hurdle**: Disassembly confirmed plausible raw E5RT event signatures. A guarded experiment retains stage A's completion event and binds it as a dependent event on stage B; both raw calls return `0`, but the process segfaults during/after the second operation prepare path. The experiment is hidden behind `--bind-e5-events` and is not the default path.

**Lesson**: The event API is real, but CoreML's event lifecycle/order matters. Directly attaching the retained completion event is too early, too late, or missing future-value bookkeeping.

**Next**: Reproduce CoreML's ObjC event lifecycle around `_bindNewCompletionEventsDirectlyWithCompletionSyncPoint:`, `_bindNewWaitEventsDirectlyWithWaitSyncPoints:`, `_updateCompletionEventFutureValuesWithCompletionSyncPoint:`, and `_updateWaitEventFutureValuesWithWaitSyncPoints:` before any correctness or performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="064-phi-stateful-raw-e5rt-chain-smoke.html">Previous: Journal 064</a> | <a href="066-phi-e5-objc-sync-point-experiment.html">Next: Journal 066</a></nav>
