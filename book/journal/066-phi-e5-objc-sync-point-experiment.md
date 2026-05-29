---
layout: default
title: "Journal 066 - Phi E5 ObjC Sync-Point Experiment"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="065-phi-public-two-call-reference-and-e5-event-probe.html">Previous: Journal 065</a> | <a href="067-phi-e5-direct-objc-event-bind-outcome.html">Next: Journal 067</a></nav>

# 2026-04-28 - Phi E5 ObjC Sync-Point Experiment

**Intent**: Test whether CoreML's `MLPredictionSyncPoint` route can express the stage-A completion / stage-B wait dependency without directly calling raw E5RT event bind functions.

**Setup**: Widened `coreml_e5_class_dump.m` to include event/sync classes. Found private `MLPredictionSyncPoint` with `initWithSharedEvent:value:` and hidden `MLPredictionOptions` accessors for `completionSyncPoint` and `waitSyncPoints`. Added `--objc-sync-points` to `e5_two_op_stream_probe`, creating an `MTLSharedEvent` and passing completion/wait sync points through the private bind methods. Added `--objc-sync-update` as a separate explicit experiment for the private future-value update hooks.

**Result**: `--objc-sync-points` is stable: raw prepare, encode, and execute all succeed, but stage B remains all zeros. `--objc-sync-update` segfaults at `_updateCompletionEventFutureValuesWithCompletionSyncPoint:` in the manual raw lifecycle, so it is off by default.

**Surprise / hurdle**: Sync-point options are accepted by `MLPredictionOptions`, but they do not repair the raw multi-op path when CoreML's normal stream preparation/async submission lifecycle is bypassed.

**Lesson**: The event dependency is probably not just data stored in options; it is created and advanced by a specific CoreML operation lifecycle. Calling the leaf update hook directly is unsafe.

**Next**: Trace or reuse `prepareForInputFeatures:options:error:` / `prepareAsyncSubmissionForInputFeatures:options:error:` ordering to learn when CoreML binds shared events, updates future values, and raw-encodes the operation.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="065-phi-public-two-call-reference-and-e5-event-probe.html">Previous: Journal 065</a> | <a href="067-phi-e5-direct-objc-event-bind-outcome.html">Next: Journal 067</a></nav>
