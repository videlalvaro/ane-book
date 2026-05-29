---
layout: default
title: "Journal 083 - Phi Dead Artifact Cleanup Outcome"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="082-phi-dead-artifact-cleanup-approval-intent.html">Previous: Journal 082</a> | <a href="084-phi-public-algorithmic-perf-direction-intent.html">Next: Journal 084</a></nav>

# 2026-04-29 - Phi Dead Artifact Cleanup Outcome

**Intent**: Reclaim disk by deleting only the approved Phi dead artifacts already ruled out by CPU fallback, slower profiling, or duplication, following measurement discipline.

**Setup**: Deleted local artifacts; generated top-k LM-head top1 s0 `.mlmodelc`/`.mlpackage`; generated 3-way and 8-way LM-head `.mlmodelc`/`.mlpackage` artifacts while preserving their manifests; and slower 5-layer tail generated artifacts [20,25), [25,30), plus duplicate [30,32).

**Result**: Cleanup completed. Disk free increased from about 6.2G to 14G; `du` reported the deletion set at 9.3G total. The current 20+4+6+2 baseline artifacts and the batch-4 LM-head set were preserved.

**Surprise / hurdle**: The main risk was avoiding accidental deletion of useful manifests or current baselines while removing large generated experiment outputs.

**Lesson**: Destructive artifact cleanup is safe when deletion targets are tied to recorded failure/slower evidence and protected baselines are named explicitly.

**Next**: Continue from the preserved 20+4+6+2 baseline and batch-4 LM-head artifacts; require separate approval for any further artifact deletion.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-models/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="082-phi-dead-artifact-cleanup-approval-intent.html">Previous: Journal 082</a> | <a href="084-phi-public-algorithmic-perf-direction-intent.html">Next: Journal 084</a></nav>
