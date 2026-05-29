---
layout: default
title: "Journal 066 - Phi Stream-Level CoreML Sync-Point Experiment"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="065-phi-public-two-call-reference-and-stream-event-probe.html">Previous: Journal 065</a> | <a href="067-phi-stream-event-bind-outcome.html">Next: Journal 067</a></nav>

# 2026-04-28 - Phi Stream-Level CoreML Sync-Point Experiment

**Intent**: Test whether CoreML's sync-point route can express the stage-A completion / stage-B wait dependency without touching lower-level stream event hooks.

**Setup**: Widened `local class-inspection probe` to include event/sync classes. Tested completion/wait sync-point plumbing through the local stream probe. The experiment stayed research-only and did not enter the public runtime.

**Result**: `--objc-sync-points` is stable: prepare, encode, and execute all succeed, but stage B remains all zeros. `--objc-sync-update` crashes in the manual lifecycle, so it is off by default.

**Surprise / hurdle**: Sync-point options are accepted, but they do not repair the multi-operation path when CoreML's normal stream preparation/async submission lifecycle is bypassed.

**Lesson**: The event dependency is probably not just data stored in options; it is created and advanced by a specific CoreML operation lifecycle. Calling the leaf update hook directly is unsafe.

**Next**: Trace or reuse `prepareForInputFeatures:options:error:` / `prepareAsyncSubmissionForInputFeatures:options:error:` ordering to learn when CoreML binds shared events, updates future values, and manual-encodes the operation.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="065-phi-public-two-call-reference-and-stream-event-probe.html">Previous: Journal 065</a> | <a href="067-phi-stream-event-bind-outcome.html">Next: Journal 067</a></nav>
