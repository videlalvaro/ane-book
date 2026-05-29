---
layout: default
title: "Experiment 24 - Structured CoT as a Grammar-Constrained Sampler"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="23-prompt-lookup-n-gram-speculation.html">Previous: Experiment 23</a> | <a href="25-prompt-lookup-force-mode-as-a-head-skip-ceiling.html">Next: Experiment 25</a></nav>

# Experiment 24 - Structured CoT as a Grammar-Constrained Sampler

**Sources**: Dechter constraint propagation + Willard/Louf guided generation +
Kaya Omer, "Structured CoT: Shorter Reasoning with a Grammar File" (2026)

The structured-CoT post is directly relevant to Phi-on-ANE, but the expected win
is energy/tokens-per-task, not raw `tok/s`. The mechanism is an inference
harness: constrain the scratchpad with a finite-state grammar such as
`GOAL/STATE/ALGO/EDGE/VERIFY`, then leave the code/answer channel permissive.

This fits the ANE-only mandate because it lives at the permitted host-side
sampling boundary. Current Phi generation already does all heavy work in ANE
layer shards plus ANE LM-head shards, then the host scans logits for argmax.
A grammar/FSM would replace unconstrained argmax with constrained argmax over
the valid next-token set. No CPU/GPU matmul, norm, attention, FFN, or LM-head
compute is introduced.

Expected benefits:

- fewer generated tokens for coding tasks with overlong scratchpads
- less answer-channel drift, empty-code output, and malformed code framing
- deterministic output sections that make downstream extraction cheaper
- compatibility with prompt-lookup/speculative work because structured outputs
  tend to repeat labels, newlines, indentation, and delimiters

ANE-specific caveats:

- It does not reduce per-token ANE compute. It saves energy only if total tokens
  drop, or if forced literals skip LM-head prediction in a future `advanceOnly`
  path.
- Phi-4-mini-instruct is not necessarily a native reasoning model; `<think>` is
  not a special single token in the local GGUF tokenizer. A Phi grammar should
  use tokenizer-aware literal sequences and short visible planning fields, not
  assume Qwen-style hidden-thinking behavior.
- Literal grammar tokens still need layer execution to populate KV cache. Forced
  labels can skip the LM-head call only if the runtime adds a safe layer-only
  state-advance path.
- Line/code fields need hard caps and metrics for comment bloat, post-plan
  bloat, syntax errors, and task pass rate. Token count alone is not enough.

Smallest implementation path:

1. Add a tokenizer-derived grammar manifest for fixed literals and newline.
2. Add constrained argmax at the existing host sampling point.
3. Add optional forced-token advance for grammar literals to avoid unnecessary
   LM-head calls while still updating KV on ANE.
4. Gate on a small coding suite: pass/fail, total tokens, plan tokens, code
   extraction errors, and energy per solved task.

Implemented first shipping slice:

- `python/phi4_mini_structured_cot_manifest.py` builds a Phi-tokenizer-aware
  manifest at `local-artifacts/phi4_mini_ane/phi4mini_structured_cot_plan.json`.
- `local-artifacts/phi4_mini_ane.swift` accepts `--structured-cot` or
  `--structured-cot-manifest <path>`.
- Deterministic grammar literals force the next token and skip LM-head
  prediction, but still run the ANE layer stack to keep KV state exact.
- Free planning fields use constrained argmax that blocks stop tokens and can
  force newline after each field's token budget.
- The default greedy path is unchanged when structured mode is not enabled.

Smoke command:

```bash
local-artifacts/phi4_mini_ane_runtime \
  --meta local-artifacts/phi4_mini_ane/phi4mini_runtime_meta_20_4_6_2.json \
  --max-new 16 \
  --structured-cot \
  --profile
```

Smoke result: structured mode forced 6 literal tokens, emitted 10 field-content
tokens, completed no fields within the short 16-token budget, and decoded at
`16.609 tok/s` after cold CoreML first-use load. Per-token decode profile stayed
in the known public baseline range: `layers_ms=56.151`,
`head_predict_reduce_ms=4.049`.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="23-prompt-lookup-n-gram-speculation.html">Previous: Experiment 23</a> | <a href="25-prompt-lookup-force-mode-as-a-head-skip-ceiling.html">Next: Experiment 25</a></nav>
