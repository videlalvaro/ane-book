---
layout: default
title: "Journal 067 - Phi Stream-Level Direct CoreML Event Bind Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="066-phi-stream-sync-point-experiment.html">Previous: Journal 066</a> | <a href="068-phi-stream-second-operation-boundary-narrowed.html">Next: Journal 068</a></nav>

# 2026-04-28 - Phi Stream-Level Direct CoreML Event Bind Outcome

**Intent**: Determine whether explicitly attaching CoreML completion/wait shared events is enough to make manual Phi stage B consume stage A's output.

**Setup**: Added an explicit event-bind experiment after manual input/output port binding.

**Result**: The direct bind hooks work structurally and attach the same shared event on both sides. Prepare, encode, and execute still return success. Stage A remains correct, but stage B remains all zeros.

**Surprise / hurdle**: Attaching the event object is not enough; the missing behavior is likely future-value update/signaling, manual encode consumption of the event state, or a separate output backing synchronization step.

**Lesson**: The dependency problem is narrower now: event objects can be attached, but CoreML's normal lifecycle does more than attach them.

**Next**: Trace `prepareAsyncSubmissionForInputFeatures:options:error:` or the normal async path to see when future values are updated and when stream runtime encode consumes event dependencies.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="066-phi-stream-sync-point-experiment.html">Previous: Journal 066</a> | <a href="068-phi-stream-second-operation-boundary-narrowed.html">Next: Journal 068</a></nav>
