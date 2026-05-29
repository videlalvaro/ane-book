---
layout: default
title: "Glossary"
---

# Glossary

Short definitions for terms used throughout the book.

## Inference Basics

**Autoregressive decode**: Generating text one token at a time, feeding each sampled token back into the next model call.

**Decode**: The one-token-at-a-time generation phase after the prompt has been processed.

**Embedding**: A learned vector looked up from a token ID before the transformer layers run.

**Hidden state**: The vector representation carried through the transformer stack.

**KV cache**: Stored key and value tensors from previous tokens, reused so decode does not recompute the entire prefix.

**Logits**: Raw scores over the vocabulary. Sampling or argmax turns logits into the next token ID.

**Prefill**: The phase that processes the prompt tokens before decode begins.

**Projection**: A learned linear map, usually written as `y = Wx`. Attention, FFNs, and LM heads are projection-heavy.

**Token**: An integer ID representing a text fragment.

## ANE and CoreML

**ANE**: Apple Neural Engine, Apple's fixed-function neural accelerator.

**ANEF**: The ANE compiler used during CoreML compilation to decide whether operations can run on the Neural Engine.

**CoreML MIL**: CoreML's Model Intermediate Language, the graph representation produced during conversion.

**`ios18.conv`**: The CoreML operation class that maps 1x1 convolution projections onto ANE.

**`MLComputePlan`**: The ground-truth API for checking which compute device CoreML selected for each operation.

**`mlmodelc`**: A compiled CoreML model directory produced by `xcrun coremlcompiler compile`.

**`mlpackage`**: A CoreML model package before compilation.

**`MLState`**: CoreML's public API for state tensors that persist across `prediction()` calls.

**RangeDim**: A CoreML shape declaration that allows a dimension, such as token length `T`, to vary within bounds at runtime.

**Residency**: Whether the intended operations actually run on ANE rather than CPU or GPU.

## Porting and Validation

**Cosine gate**: A quality check comparing CoreML output to a reference output with cosine similarity, usually requiring at least `0.97`.

**Golden**: A trusted reference output captured from a known-good backend, usually PyTorch or FP16 CoreML.

**Shard**: A separately compiled piece of a larger model, such as a few transformer layers or one LM-head slice.

**Silent fallback**: A failure mode where a model compiles and runs correctly but CoreML places important operations on CPU or GPU instead of ANE.

## Model Architecture

**Attention**: The transformer mechanism that lets a token read earlier tokens using query, key, and value projections.

**FFN**: Feed-forward network inside a transformer block, usually the largest projection-heavy part of a dense layer.

**LM head**: The final projection from hidden state to vocabulary logits.

**MoE**: Mixture of Experts, a layer design with multiple expert FFNs and a router that chooses which experts contribute.

**RMSNorm**: A normalization layer commonly used in modern decoder-only LLMs.
