---
layout: default
title: "Journal 096 - Phi-4-mini T=4 Verifier Scale-Out Intent"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="095-phi-4-mini-real-weight-t-4-verifier-layer-passed.html">Previous: Journal 095</a> | <a href="097-phi-4-mini-t-4-verifier-scale-out-outcome.html">Next: Journal 097</a></nav>

# 2026-04-29 - Phi-4-mini T=4 Verifier Scale-Out Intent

**Intent**: Move from the one-layer real-weight `T=4` verifier pass to export plumbing for the production speculative-verifier topology, while preserving the public CoreML/ANE-only boundary. This follows Experiment 26: Dragon Book data-flow invariants for block-vs-sequential equivalence, Knuth sequential verification for exact accept/reject semantics, and Leviathan et al. (2023) speculative decoding framing. Checkpoint anchors are `phi4-mini-ane-v0-spec-2026-04-29` at `b366672` and `phi4-mini-t4-layer0-pass-2026-04-29` at `290e3d3`.

**Setup**: Planning note before implementation. Target topology is the production public CoreML path with `T=4` multi-layer shards and manifest references, aligned with the existing `20+4+6+2` layer layout. Build order starts with the smallest tail shard first, expected tail range `[30,32)`, to minimize disk/RAM risk before larger verifier shards. Disk is tight at roughly 11 GiB free and existing artifacts are large, so no `.mlpackage`, `.mlmodelc`, `.npz`, helper script, `models/`, or other large artifact cleanup may occur without explicit user confirmation.

**Result**: Intent recorded before scale-out. No exporter changes, manifest changes, T=4 multi-layer artifacts, placement counts, parity numbers, latency, energy, cosine, perplexity, cleanup, or deletion have been run for this entry.

**Surprise / hurdle**: The one-layer verifier passed real-weight parity and strict residency, but scale-out now has two independent hazards: exact block-vs-four-single-token semantics across fused ranges, and tight disk headroom that makes accidental full-scale artifact generation or cleanup especially costly.

**Lesson**: T=4 verifier scale-out should start from the smallest production shard and advance only through parity plus strict ANE residency gates; disk pressure is a scheduling constraint, not permission for unconfirmed destructive cleanup or CPU/GPU fallback.

**Next**: Add the `T=4` multi-layer shard exporter and runtime manifest references; build/compile the smallest tail shard first; validate block-vs-sequential parity and strict MLComputePlan residency before any larger shard or full scale-out. Keep the path public CoreML only, with no private API and no CPU/GPU compute fallback.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="095-phi-4-mini-real-weight-t-4-verifier-layer-passed.html">Previous: Journal 095</a> | <a href="097-phi-4-mini-t-4-verifier-scale-out-outcome.html">Next: Journal 097</a></nav>
