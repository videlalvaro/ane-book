---
layout: default
title: "Journal 057 - Stream Dispatch Bridge Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="056-stream-dispatch-investigation-intent.html">Previous: Journal 056</a> | <a href="058-coreml-stream-bridge-operation-handles-recovered.html">Next: Journal 058</a></nav>

# 2026-04-28 - Stream Dispatch Bridge Outcome

**Intent**: Record the outcome of the stream dispatch investigation after checkpointing the public Phi-4-mini ANE runtime state, following call-hoisting/strength-reduction discipline for reducing CoreML shard boundary costs.

**Setup**: Checkpoint/tag `phi4-mini-ane-q8-fusion-17tok-2026-04-28` was created on commit `f273a47` before investigation. Local probes inspected whether the public CoreML load path exposed enough stream-level structure to reduce shard-boundary overhead.

**Result**: The stream-level boundary was observable, but the direct probe still failed on current Phi public-CoreML shards at a legacy compiled-artifact contract. The useful lesson was that public CoreML loading already creates a program-library object that can explain why shard-boundary overhead exists.

**Surprise / hurdle**: The public CoreML stream runtime path already owns a usable program-library handle, while the direct stream dispatch probe is blocked by older CoreML execution runtime artifact expectations that public `.mlmodelc` shards do not satisfy.

**Lesson**: The next unsupported path should investigate the CoreML stream runtime handle/operation bridge rather than trying to synthesize legacy CoreML execution runtime artifacts first.

**Next**: Follow the stream runtime program-library handle toward operation/chaining surfaces for already-loaded CoreML models; keep public MLComputePlan residency and golden validation as acceptance gates before any performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="056-stream-dispatch-investigation-intent.html">Previous: Journal 056</a> | <a href="058-coreml-stream-bridge-operation-handles-recovered.html">Next: Journal 058</a></nav>
