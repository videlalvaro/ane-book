---
layout: default
title: "Journal 057 - Private ANE API Bridge Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="056-private-ane-chaining-investigation-intent.html">Previous: Journal 056</a> | <a href="058-coreml-e5-bridge-operation-handles-recovered.html">Next: Journal 058</a></nav>

# 2026-04-28 - Private ANE API Bridge Outcome

**Intent**: Record the outcome of the private ANE API investigation after checkpointing the public Phi-4-mini ANE runtime state, following call-hoisting/strength-reduction discipline for reducing CoreML shard boundary costs.

**Setup**: Checkpoint/tag `phi4-mini-ane-q8-fusion-17tok-2026-04-28` was created on commit `f273a47` before investigation. Probes included direct `_ANEClient` / `prepareChaining` selector inspection, `ane_chain_probe` against current Phi public-CoreML `.mlmodelc` shards, and new `ane_coreml_bridge_probe.m` to inspect the public CoreML load path.

**Result**: Direct `_ANEClient` and `prepareChaining` selectors are present. `ane_chain_probe` still fails on current Phi public-CoreML shards at the legacy Espresso contract because `model.espresso.net` is missing. `ane_coreml_bridge_probe.m` shows public CoreML `MLModel` load registers a model UUID in `_ANEClient connectionsUsedForLoadingModels` and exposes the chain `MLDelegateModel -> MLE5Engine -> MLE5ProgramLibrary -> e5rt_program_library` handle. `_programLibraryHandleWithForceRespecialization:error:` returned non-null with no error.

**Surprise / hurdle**: The public CoreML E5RT path already owns a usable program-library handle, while the direct private chaining probe is blocked by older Espresso artifact expectations that public `.mlmodelc` shards do not satisfy.

**Lesson**: The next private path should investigate the CoreML E5RT handle/operation bridge rather than trying to synthesize legacy Espresso artifacts first.

**Next**: Follow the E5RT program-library handle toward operation/chaining surfaces for already-loaded CoreML models; keep public MLComputePlan residency and golden validation as acceptance gates before any performance claim.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="056-private-ane-chaining-investigation-intent.html">Previous: Journal 056</a> | <a href="058-coreml-e5-bridge-operation-handles-recovered.html">Next: Journal 058</a></nav>
