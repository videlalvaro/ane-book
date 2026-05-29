---
layout: default
title: "Diagram Plan"
---

# Diagram Plan for The Apple Neural Engine Inference Book

The book needs visuals that make the ANE mental model concrete: tensor shapes,
compile gates, shard topology, KV state, and speculative decoding. Prefer clean
SVGs checked into `book/assets/diagrams/`; use lightweight animated SVG only
where motion clarifies time, state, or repeated dispatch.

## Asset Style

- Use a restrained engineering style: white or near-white background, dark text,
  one accent color for ANE, one for CPU/GPU fallback, one for host/runtime.
- Keep each diagram narrow enough for mobile: max width around 900 px, with text
  labels readable at 360 px.
- Put the concept in the visual, not in decorative art. These diagrams should
  explain shapes, residency, flow, and tradeoffs.
- Prefer SVG for all static diagrams. Use animated SVG for state/time flows such
  as decode loops, KV writes, RangeDim T changes, and speculative acceptance.
- Store source and exported assets together if using a generator:
  `book/assets/diagrams/<chapter>/<name>.svg` and optionally
  `book/assets/diagrams/<chapter>/<name>.mmd`.

## Chapter 0: Modern Inference and Why ANE?

### 0.0 Modern Inference Primer

Insert before `The Hardware You're Not Using`.

Diagrams implemented:

- `autoregressive-decode-loop.svg`: tokenization, embeddings, transformer, logits,
  sampling, and feedback.
- `prefill-vs-decode.svg`: prompt block processing vs one-token decode.
- `transformer-block-anatomy.svg`: RMSNorm, attention, FFN, and residual flow.
- `where-matmul-appears.svg`: Q/K/V/O, FFN, and LM-head projection hotspots.
- `kv-cache-concept.svg`: append-only K/V rows and prefix reads.

Format: static SVG.

Priority: implemented. This is the didactic ramp before ANE-specific rules.

### 0.1 Hardware Tradeoff Overview

Insert after `ANE vs GPU vs CPU - the Real Tradeoffs`.

Diagram: three-column visual comparing ANE, GPU, and CPU as inference engines.
Show power, programmability, and best-fit workloads as compact visual badges.

Format: static SVG.

Priority: high. This is the first place the book can stop feeling like a table
and start giving the reader a mental picture.

### 0.2 CoreML Dispatch Pipeline

Insert after `What CoreML Actually Does`.

Diagram: PyTorch/TorchScript -> CoreML mlprogram -> coremlcompiler -> `.mlmodelc`
-> runtime dispatch to ANE/GPU/CPU. Highlight that the placement decision is made
at compile time and verified later with `MLComputePlan`.

Format: static SVG.

Priority: high.

### 0.3 Matmul as 1x1 Conv

Insert inside `Why Every Matmul Must Be Conv2d`, immediately before or after the
code block.

Diagram: left side `W [C_out, C_in] x x [C_in]`; right side equivalent
`Conv2d [C_out, C_in, 1, 1]` over activation `[1, C_in, T, 1]`. Include the
output shape `[1, C_out, T, 1]`.

Format: static SVG, with optional subtle animated variant where a token column
slides through the 1x1 kernel.

Priority: must-have. This is the central trick of the whole book.

## Chapter 1: ANE Empirical Laws

### 1.1 Canonical Tensor Shape

Insert after `Law 1: The Shape Must Be [1, channels, T, 1]`.

Diagram: labeled 4D tensor box showing batch=1, channels=C, height=T, width=1.
Show bad shapes falling to CPU/GPU and the canonical shape landing on ANE.

Format: static SVG.

Priority: must-have.

### 1.2 Residency Gate

Insert after `Law 6: MLComputePlan Is the Ground Truth`.

Diagram: compile success is not enough. Show two paths from `.mlmodelc`: green
path where `ios18.conv` ops prefer Neural Engine, red path where the model still
runs but conv ops prefer CPU.

Format: static SVG.

Priority: high.

### 1.3 Shard Size Cliff

Insert after `Law 4: The Shard Size Ceiling Is ~250 MB`.

Diagram: horizontal ruler from 0 MB to 1 GB with zones: too small can fall to CPU
for some patterns, validated range, risky 250 MB cliff, error -14 region.

Format: static SVG.

Priority: high.

### 1.4 Chain Primitive Concept

Insert after `Law 7: The Chain Primitive Enables Zero-Copy Multi-Stage Dispatch`.

Diagram: stage 0 -> stage 1 -> stage 2 with loopback symbol indexes and shared
memory pool. Contrast with host round-trip copies.

Format: animated SVG if possible: pulse an activation buffer through stages
without leaving the ANE memory pool.

Priority: medium. It is advanced, but the text is abstract without a picture.

## Chapter 2: Porting Recipe

### 2.1 End-to-End Porting Flow

Insert near the top of the chapter, after the intro paragraph.

Diagram: GGUF -> metadata extraction -> Conv2d graph -> weight loading -> CoreML
conversion -> INT8 quantization -> compile -> residency gate -> golden gate ->
Swift runtime.

Format: static SVG, likely a numbered pipeline.

Priority: must-have. This chapter is a recipe and needs a map.

### 2.2 Weight Shape Transformation

Insert after `Step 2: Load Weights from GGUF`.

Diagram: GGUF Q8_0 blocks -> dequantized FP32 matrix -> reshaped Conv2d weight
`[out, in, 1, 1]` -> quantized CoreML weight blob.

Format: static SVG.

Priority: high.

### 2.3 Conversion Environment Split

Insert after the three Python environments table.

Diagram: three lanes for `.venv`, `.venv313`, and Xcode `python3`, each with its
allowed task. Add warning edge for mixing coremltools versions.

Format: static SVG.

Priority: medium. This prevents a real workflow mistake.

### 2.4 Validation Gates

Insert before `Summary Checklist`.

Diagram: two gate doors: `MLComputePlan == 100% ANE` and `cosine >= 0.97`; only
then does benchmarking unlock.

Format: static SVG.

Priority: high.

## Chapter 3: Quantization

### 3.1 Quantization Family Map

Insert after `The Only Safe Baseline: INT8 Per-Tensor`.

Diagram: branch map of INT8 per-tensor, INT4 per-block, INT4 palettization,
W8A8. Annotate each with status: production, dangerous fallback, promising,
future.

Format: static SVG.

Priority: must-have.

### 3.2 Silent CPU Fallback

Insert after `The INT4 Per-Block Silent CPU Fallback` reproduction table.

Diagram: same compiled model, two placements: INT8 conv fused onto ANE; INT4
per-block dequant pattern split into lookup/scale/indexing and rejected to CPU.

Format: static SVG.

Priority: high.

### 3.3 Per-Tensor vs Per-Block vs Palettized Weights

Insert after `Why INT8 Works` or `INT4 Palettization`.

Diagram: three tiny weight matrices showing one global scale, block scales, and
codebook lookup. Include how each maps to ANE friendliness.

Format: static SVG.

Priority: high.

### 3.4 Norm Convention Bug

Insert after `The Norm Convention Bug: A Case Study`.

Diagram: split path comparing correct RMSNorm `(1 + gamma)` vs wrong `gamma`,
then show shifted logits/cosine drop. This can be a caution-card visual.

Format: static SVG.

Priority: medium.

## Chapter 4: Shard Sizing

### 4.1 250 MB Wall

Insert after `The 250 MB Wall` data table.

Diagram: cliff chart: shard size on x-axis, outcome on y-axis or color band.
Place Phi, Gemma, ZAYA, Qwen examples as labeled markers.

Format: static SVG.

Priority: must-have.

### 4.2 Layer Packing Calculator

Insert after `Layer Counting: How Many Layers Per Shard?`.

Diagram: per-layer components stacked as blocks: QKV/O, FFN gate/up/down, norms.
Show how layer MB accumulates into a shard until the safe limit.

Format: static SVG.

Priority: high.

### 4.3 LM Head Slicing

Insert after `The LM Head Problem`.

Diagram: large `[vocab, d_model]` LM head split horizontally into shard slices,
each producing a logit slice; host concatenates and samples.

Format: static SVG.

Priority: must-have.

### 4.4 Runtime Artifact Layout

Insert after `Shard Naming Convention`.

Diagram: folder tree plus arrows into `runtime_meta.json`, showing how the Swift
runtime discovers layers, LM head shards, and embeddings.

Format: static SVG.

Priority: medium.

## Chapter 5: Stateful KV Cache

### 5.1 Decode Loop with State

Insert after the first decode-loop code block.

Diagram: token -> embedding -> layer stack with state -> LM head -> sample ->
next token. Show state as a sidecar attached to every layer call.

Format: animated SVG. Pulse a token through the loop, with the sampled token
feeding back.

Priority: must-have.

### 5.2 KV Cache Write

Insert after the scatter/write example.

Diagram: K/V cache as a long matrix with positions. Highlight current `pos`,
write `k_new/v_new`, and attention reading `0..pos`.

Format: animated SVG if possible, with the write cursor advancing.

Priority: must-have.

### 5.3 Prefill to Decode Handoff

Insert after `The CCA Pattern` prefill/decode code block.

Diagram: chunked prefill T=4 writes multiple KV slots per call; decode T=1 then
continues from the next position.

Format: animated SVG.

Priority: high.

### 5.4 KV Memory Budget

Insert after the KV memory formula.

Diagram: stacked memory blocks per layer, K and V, heads, sequence length. Show
how `max_seq_len` changes total memory.

Format: static SVG.

Priority: medium.

## Chapter 6: RangeDim and Speculative Decode

### 6.1 Prefill vs Decode Shape

Insert after `The Prefill/Decode Asymmetry`.

Diagram: T=N prefill as a wide activation image and T=1 decode as a single
column. Show compute-bound vs bandwidth-bound labels.

Format: static SVG.

Priority: must-have.

### 6.2 RangeDim Specialization

Insert after `RangeDim: Variable Sequence Length in CoreML`.

Diagram: one `.mlmodelc` with four runtime specializations T=1,2,3,4. Show first
use JIT specialization as a small compile/warmup badge.

Format: animated SVG optional: width changes from T=1 to T=4.

Priority: high.

### 6.3 n-Gram Draft and Verify

Insert before or after the speculative decode pseudocode.

Diagram: history tokens feed n-gram lookup, draft tokens enter a T=4 verifier,
green prefix accepted, red first mismatch falls back to target token.

Format: animated SVG. This is one of the best animation candidates in the book.

Priority: must-have.

### 6.4 Stateful RangeDim Failure Mode

Insert after `Stateful + RangeDim: Known Complication`.

Diagram: correct T=1 writes one slot; risky T>1 writes multiple slots. Show a bad
dynamic slice writing to the wrong address and the validator catching divergence.

Format: static SVG.

Priority: medium.

## Chapter 7: Mixture of Experts on ANE

### 7.1 MoE Routing Overview

Insert after `Why MoE on ANE Is Hard`.

Diagram: router -> top-k expert selection -> expert FFNs -> weighted sum. Then
mark the conditional dispatch part as the hard problem for CoreML.

Format: static SVG.

Priority: must-have.

### 7.2 Soft Routing vs Sparse Routing

Insert after `Approach 1: Soft Routing`.

Diagram: side-by-side. Soft routing runs all experts with zero/low weights;
sparse routing runs only top-k experts. Include compute cost labels.

Format: static SVG, optional animation showing all experts lighting up vs only
two experts lighting up.

Priority: must-have.

### 7.3 ZAYA Shard Layout

Insert after `ZAYA1-8B Architecture` shard tree.

Diagram: alternating attention and MoE shards, then split LM head, then host
embedding. Use colors for attention, MoE, LM head, host.

Format: static SVG.

Priority: high.

### 7.4 Packed Expert Shard

Insert after `Packing All Experts Into One Shard`.

Diagram: one MoE `.mlmodelc` containing router plus 8 expert FFNs, all unrolled
into conv ops. Show `router_weights[:, i]` gating each expert output.

Format: static SVG.

Priority: high.

### 7.5 Privacy Filter Output Flow

Insert after `Privacy Filter: MoE NER`.

Diagram: text tokens -> MoE NER -> BIO labels -> spans -> redacted text.

Format: static SVG.

Priority: medium. It lightens the chapter with a practical non-chat example.

## Chapter 8: Experiment Index

Chapter 8 is long and should not get a diagram for every experiment. Add section
breakers that summarize clusters of experiments visually.

### 8.1 Experiment Timeline

Insert near the start of `Phi ANE Shape Optimization Program`.

Diagram: timeline from Exp 16 to Exp 36 grouped into themes: EML ideas, Phi
shape search, speculative decode, MicroGPT floor, HyMT, ZAYA, Gemma quality.

Format: static SVG.

Priority: must-have for readability.

### 8.2 Phi Topology Search

Insert around Experiment 20.

Diagram: weighted path over layer indices `0..32`, showing candidate shard edges
and selected `20+4+6+2` topology.

Format: static SVG.

Priority: high.

### 8.3 T=4 Verifier Block

Insert around Experiment 26.

Diagram: verifier input shapes `x`, RoPE tables, attention mask, KV write mask,
and output hidden/logits for T=4. Show append-only KV slots.

Format: animated SVG if possible.

Priority: must-have.

### 8.4 RangeDim Unification

Insert around `RangeDim unification`.

Diagram: before: separate T=1 and T=4 shard sets exceed ANE memory. After: one
RangeDim shard JIT-specializes for T=1 and T=4.

Format: static SVG.

Priority: high.

### 8.5 Prompt-Length Speedup Curve

Insert after the prompt-length sweep table.

Diagram: simple line chart of decode speedup versus prompt length. Use the table
values 100, 200, 372, 800 tokens.

Format: static SVG generated from data.

Priority: high.

### 8.6 MicroGPT Minimum Size Floor

Insert around Experiment 27.

Diagram: tiny 0.03 MB graph falls to CPU; scaled 19.07 MB graph lands on ANE.
Show the empirical 14 MB floor.

Format: static SVG.

Priority: high.

### 8.7 ZAYA Attention/MoE Cost Split

Insert around Experiments 29-35.

Diagram: stacked per-token cost bar showing attention, MoE, LM head. Then show
why T=4 helps attention more than soft-routed MoE.

Format: static SVG.

Priority: must-have.

### 8.8 Quantization Quality Funnel

Insert around Experiment 36.

Diagram: per-tensor INT8 -> per-layer cosine spread -> compounded full-stack
error -> per-channel rebuild target. Include the `cos 0.9555` to `>=0.997`
story as a quality funnel.

Format: static SVG.

Priority: high.

## Chapter 9: Decision Journal

Chapter 9 is an audit trail. Diagrams should be sparse and should summarize
patterns across entries rather than decorate individual notes.

### 9.1 Decision Timeline by Theme

Insert near the top of the journal.

Diagram: horizontal timeline with color-coded swimlanes: Phi shard search, E5
private API, public runtime, speculative decode, HyMT/ZAYA/Gemma. This gives
readers a way to orient before the dense journal entries.

Format: static SVG.

Priority: high.

### 9.2 Phi Shard Evolution

Insert around the Phi entries before the private E5 investigation.

Diagram: progression from 1-layer shards to 3-layer, 4-layer, 20+4+6+2. Use
small bars over a 32-layer ruler.

Format: static SVG.

Priority: medium.

### 9.3 E5 Private API Investigation Map

Insert around the E5 chain entries.

Diagram: explored path map: ObjC reflection -> operation handles -> event bind
-> raw memory bridge -> one-stream timing reality check. Mark the final public
decision boundary.

Format: static SVG.

Priority: medium.

### 9.4 Validation-First Loop

Insert near the end or as a journal preface.

Diagram: hypothesis -> build shard -> residency gate -> golden gate -> benchmark
-> keep/revert. This explains the rhythm of the journal.

Format: static SVG.

Priority: high.

## First Implementation Batch

Build these first because they will improve the reader experience the most:

1. Chapter 0: Matmul as 1x1 Conv.
2. Chapter 2: End-to-end porting flow.
3. Chapter 4: LM head slicing.
4. Chapter 5: KV cache write animation.
5. Chapter 6: n-gram draft and verify animation.
6. Chapter 7: soft routing vs sparse routing.
7. Chapter 8: experiment timeline.

## Suggested File Layout

```text
book/
  assets/
    diagrams/
      00-why-ane/
        matmul-as-conv2d.svg
        coreml-dispatch-pipeline.svg
      02-porting-recipe/
        porting-flow.svg
      05-stateful-kv-cache/
        kv-cache-write.svg
      06-rangedim-speculative/
        ngram-draft-verify.svg
      07-moe-on-ane/
        soft-vs-sparse-routing.svg
      08-experiments/
        experiment-timeline.svg
```

## Implementation Notes

- Animated SVG should use CSS keyframes inside the SVG, not JavaScript, so GitHub
  Pages can serve it without security exceptions.
- Keep alt text close to each image in Markdown. The visual should not be the
  only source of technical truth.
- Use Markdown image links for static diagrams:
  `![Matmul as 1x1 Conv2d](assets/diagrams/00-why-ane/matmul-as-conv2d.svg)`.
- For animations, include a one-sentence caption that describes the motion for
  readers who cannot see it.