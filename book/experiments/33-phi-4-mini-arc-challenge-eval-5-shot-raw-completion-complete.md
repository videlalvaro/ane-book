---
layout: default
title: "Experiment 33 - Phi-4-mini ARC-Challenge Eval (5-shot, raw completion) [COMPLETE]"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="32-zaya1-8b-speculative-decode-t-4-verifier-n-gram-implemented-bottlenecked.html">Previous: Experiment 32</a> | <a href="34-zaya1-8b-moe-rangedim-rebuild-t-1-4-speculative-moe-complete.html">Next: Experiment 34</a></nav>

# Experiment 33 - Phi-4-mini ARC-Challenge Eval (5-shot, raw completion) [COMPLETE]

**Date**: 2026-05-13  
**Objective**: Measure Phi-4-mini ANE accuracy on ARC-Challenge (1172-item test set,
5-shot few-shot) with the correct completion-style prompting.

### Bug discovered and fixed (v3 → v4)

`eval/models/phi4_mini_server.py` was wrapping every prompt with
`build_phi_chat_prompt(prompt_text, _SYSTEM)`, injecting
`<|system|><|end|><|user|>…<|end|><|assistant|>` markers around the already-formatted
5-shot ARC prompt.  This causes the model to answer in chat/assistant mode rather than
directly completing `"Answer: ___"`.

Effect: first ~130 items (easy, unambiguous) score ~65%; after item 130 the model
collapses to predicting `'C'` on almost every item (chat mode with a systematic bias),
final v3 accuracy **22.6%**.

Fix: removed the chat-template call; the server now tokenises `prompt_text` directly:

```python
# Before (broken):
full_prompt = build_phi_chat_prompt(prompt_text, _SYSTEM)
prompt_ids  = tokenizer.encode(full_prompt)

# After (correct):
prompt_ids = tokenizer.encode(prompt_text)   # raw 5-shot completion
```

### Results

| Run | Prompt mode | Correct / Total | Accuracy |
|-----|-------------|-----------------|----------|
| v3 (broken) | chat-template wrapped | 265 / 1172 | 22.6% |
| **v4 (fixed)** | **raw 5-shot completion** | **765 / 1172** | **65.3%** |

Prediction distribution v4 (diverse A/B/C/D throughout): no collapse observed.
Throughput: ~6–7 s/item (rangedim T=1..4 chunked prefill, 100% ANE).

### Comparison with published baselines

Phi-4-mini-instruct reported **58.7%** on ARC-Challenge (0-shot) in the Microsoft
technical report; 5-shot completion mode on our ANE runtime gives **65.3%**, consistent
with the expected few-shot uplift.

### Artifacts

- `eval/results/arc_challenge_v4.log` — full 1172-item run log
- `eval/results/phi4_mini_arc_challenge_20260513_030918.json` — JSON result record
- `eval/models/phi4_mini_server.py` — fixed (chat-wrap removed)

**Reference**: [TAOCP Vol.2 §3.2] — sampling / prediction distribution analysis;
[EoP §1] — correctness precedes performance; project policy §Quality gate.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="32-zaya1-8b-speculative-decode-t-4-verifier-n-gram-implemented-bottlenecked.html">Previous: Experiment 32</a> | <a href="34-zaya1-8b-moe-rangedim-rebuild-t-1-4-speculative-moe-complete.html">Next: Experiment 34</a></nav>
