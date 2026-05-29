---
layout: default
title: "Journal 067 - Phi E5 Direct ObjC Event Bind Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="066-phi-e5-objc-sync-point-experiment.html">Previous: Journal 066</a> | <a href="068-phi-e5-second-operation-boundary-narrowed.html">Next: Journal 068</a></nav>

# 2026-04-28 - Phi E5 Direct ObjC Event Bind Outcome

**Intent**: Determine whether explicitly attaching CoreML completion/wait shared events is enough to make raw Phi stage B consume stage A's output.

**Setup**: Added `--objc-sync-bind-direct`, which calls `_bindNewCompletionEventsDirectlyWithCompletionSyncPoint:` on stage A and `_bindNewWaitEventsDirectlyWithWaitSyncPoints:` on stage B after manual input/output port binding.

**Result**: The direct bind hooks work structurally. Stage A's `completionSharedEventBoundToESOP` and stage B's `waitSharedEventsBoundToESOP` contain the same `_MTLSharedEvent`. Raw prepare, encode, and execute still return success. Stage A remains correct, but stage B remains all zeros.

**Surprise / hurdle**: Attaching the event object is not enough; the missing behavior is likely future-value update/signaling, raw encode consumption of the event state, or a separate output backing synchronization step.

**Lesson**: The dependency problem is narrower now: event objects can be attached, but CoreML's normal lifecycle does more than attach them.

**Next**: Trace `prepareAsyncSubmissionForInputFeatures:options:error:` or the normal async path to see when future values are updated and when E5RT encode consumes event dependencies.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="066-phi-e5-objc-sync-point-experiment.html">Previous: Journal 066</a> | <a href="068-phi-e5-second-operation-boundary-narrowed.html">Next: Journal 068</a></nav>
