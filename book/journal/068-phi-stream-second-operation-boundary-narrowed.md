---
layout: default
title: "Journal 068 - Phi Stream-Level Second-Operation Boundary Narrowed"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="067-phi-stream-event-bind-outcome.html">Previous: Journal 067</a> | <a href="069-phi-stream-memory-bridge-breakthrough.html">Next: Journal 069</a></nav>

# 2026-04-28 - Phi Stream-Level Second-Operation Boundary Narrowed

**Intent**: Separate generic manual stream-level chaining problems from Phi-specific second-operation failure.

**Setup**: Added true FP16 Torch controls, a 4D `[1,C,1,1]` toy mode, a tiny `ct.StateType` stateful toy, logical-stride `MLMultiArray` stats, and a restored direct-binding mode. Added `--rebind-second-x` to re-apply hidden-to-input direct binding after CoreML sets up state/inout ports.

**Result**: Tiny controls pass: FP16 2D, FP16 4D, and FP16 4D stateful toy all chain correctly with stage A sum `14` and stage B sum `28`. Real Phi one-layer chains still fail as second op: `23_24 -> 24_25` and `24_25 -> 25_26` encode successfully, stage A is nonzero, and stage B is all zero. The original `16_24 -> 24_30` path also remains zero. Rebinding the second input and attaching a shared event did not change the zero-output result.

**Surprise / hurdle**: The bug is not Float16, 4D tensor layout, simple state, duplicate state names, direct input binding being cleared, or missing visible CoreML shared-event attachment.

**Lesson**: The failure boundary is now “real Phi stateful CoreML program as second manually encoded stream-level operation.” CoreML’s normal encoder is probably adding an inout/state dependency or scheduling relationship that the manual manual encode path still lacks.

**Next**: Trace normal `prepareAsyncSubmissionForInputFeatures:options:error:` / `prepareForInputFeatures:options:error:` around a real Phi shard and compare operation state/handles before manual encode, especially state/inout memory object binding and dependency counts.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="067-phi-stream-event-bind-outcome.html">Previous: Journal 067</a> | <a href="069-phi-stream-memory-bridge-breakthrough.html">Next: Journal 069</a></nav>
