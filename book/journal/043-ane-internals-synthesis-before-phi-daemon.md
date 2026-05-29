---
layout: default
title: "Journal 043 - ANE Internals Synthesis Before Phi Daemon"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="042-phi-4-mini-isolated-warm-cache-outcome.html">Previous: Journal 042</a> | <a href="044-phi-4-mini-resident-serve-mode-landed.html">Next: Journal 044</a></nav>

# 2026-04-28 - ANE Internals Synthesis Before Phi Daemon

**Intent**: Analyze [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md) before starting Phi daemon work, to ground the next runtime direction in observed ANE compile/load/store/runtime architecture rather than process-name inference alone.

**Setup**: Planning/synthesis only. Source reviewed: external `ane-internals` README. Findings were saved to session/repo memory. No CoreML conversion, residency validation, golden validation, performance run, cleanup, deletion, or energy benchmark was performed.

**Result**: The README describes a real architecture around `ANECompiler.framework`, `AppleNeuralEngine.framework`, `ANECompilerService.xpc`, `ANEStorageMaintainer.xpc`, `aned`, and `aneuserd`. The daemon protocol includes compile/load/cache/purge/chaining methods; compiler service behavior is path-, sandbox-, and cache-oriented. The compiler pipeline includes validation, ZinIr optimization, MIR pressure-based splitting/fusion, DMA/cache planning, register allocation/spilling, scheduling, and latency modeling. Passive process sampling is too weak as proof of ANE execution.

**Surprise / hurdle**: Private `_ANEClient`/XPC details are useful research context but should not become the production Phi path; public CoreML plus MLComputePlan remains the shippable residency proof surface.

**Lesson**: Treat ANE execution as a resident compiled-artifact lifecycle, not just a CoreML call, and prove execution with public residency/quality gates rather than daemon observation alone.

**Next**: Build a resident warm Phi daemon first, then probe ANE-side LM-head top-k/argmax because TopK/Reduction validators exist; keep private `_ANEClient`/XPC exploration separate as research.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="042-phi-4-mini-isolated-warm-cache-outcome.html">Previous: Journal 042</a> | <a href="044-phi-4-mini-resident-serve-mode-landed.html">Next: Journal 044</a></nav>
