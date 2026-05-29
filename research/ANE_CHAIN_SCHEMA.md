# ANE Execution Model Notes

**Date**: 2026-04-22, revised 2026-05-29

This note records public-safe conclusions from black-box CoreML experiments and
runtime observation. It deliberately avoids unsupported Apple symbols, framework
names, method names, probe source, and direct-driver details.

The runnable code in this repository uses public CoreML APIs only: `MLModel`,
`MLState`, `MLComputePlan`, `MLMultiArray`, compiled `.mlmodelc` artifacts, and
coremltools-generated `.mlpackage` models. These notes explain the execution
model assumptions that shaped the public implementation; they are not an API
guide and are not required to run the examples.

## What We Learned

The ANE path behaves like a compiled, procedure-oriented execution system. The
public CoreML layer hides the lower scheduling machinery, but the practical
rules show through in repeatable ways:

- Compile-time graph shape matters more than Python-level model structure.
- Static tensor graphs are much easier to keep ANE-resident than data-dependent
  control flow.
- Host round trips between tiny shards are expensive enough to shape the whole
  runtime design.
- CoreML-managed state is the supported way to keep decode state close to the
  compiled model.
- Compiled artifact layout and size can determine whether a model is accepted,
  even when the math and parameter count look reasonable.
- Placement must be verified with `MLComputePlan`; a successful compile is not
  enough.

The engineering lesson is simple: treat CoreML as the supported contract and
design graphs that make the compiler's job boring.

## Public Implementation Boundary

The production path in this repo is intentionally conservative:

1. Export static CoreML programs with coremltools.
2. Use 4D tensor layouts that map projections to `ios18.conv`.
3. Keep decode KV cache in `MLState` rather than ferrying large tensors through
   Swift on every token.
4. Split models into shards that compile below the practical package-size
   ceiling.
5. Load CoreML models once and reuse `MLState`, `MLMultiArray`, masks, and RoPE
   tables in the Swift runtime.
6. Validate every shipping shard with `MLComputePlan` before treating a number
   as an ANE number.

No checked-in converter, validator, or runtime calls unsupported Apple frameworks,
requires unsupported entitlements, or talks directly to an ANE driver.

## Dispatch and Multi-Stage Execution

Several experiments pointed at the same shape of problem: modern LLM inference
often wants to run many small compiled stages in a tight sequence, while the
public CoreML API exposes each model invocation as a host-visible call.

That observation leads to two public design rules:

- Prefer fewer, fatter shards when they still compile and stay ANE-resident.
- When a model family requires many shards, make the Swift host path allocation
  free and predictable: preallocate arrays, reuse buffers, and avoid rebuilding
  feature dictionaries or masks in the hot loop.

For MoE models, the same pressure appears as an expert-dispatch problem. Sparse
MoE wants to run only the selected experts per token, but public CoreML graphs
are easiest to place when the expert computation is static. The book therefore
ships the public, static approach: pack the experts for a layer into one MoE
shard and use soft routing or top-k-masked weighted sums inside the graph.

Future public CoreML multi-function support may make sparse expert dispatch more
ergonomic. Until that path is complete, this repo treats sparse per-token expert
dispatch as research, not production code.

## Artifact Layout Lessons

CoreML's public compiler output is the only layout this repo relies on. A few
negative controls taught useful boundaries:

- Rewriting package internals is not a robust way to make a model executable.
- Different compiled artifact families can expose different internal layouts,
  but those layouts are not a portable contract.
- A package that runs through the public CoreML loader is still not guaranteed
  to satisfy lower-level, unsupported assumptions.

The practical conclusion is to avoid depending on artifact internals. Use
coremltools, `coremlcompiler`, and `MLComputePlan` as the public contract.

## Quantization Notes

Two quantization families behaved differently:

- INT8 per-tensor linear quantization is the production baseline. It repeatedly
  kept projection-heavy shards on ANE while preserving acceptable quality.
- Linear INT4 per-block quantization caused placement failures in representative
  shards because the runtime dequantization pattern did not stay on ANE.
- LUT/palettized INT4 is a separate path from linear per-block INT4. When the
  centroids are baked into the graph in an ANE-friendly way, it can pass
  residency and quality gates for selected probes.

The rule is not "INT4 is bad." The rule is: verify the exact quantization graph
with `MLComputePlan` and golden-output comparisons before scaling it.

## How These Notes Feed the Book

These notes support the public chapters in four ways:

- Chapter 1 turns repeated placement results into empirical laws.
- Chapter 4 uses package-size failures to motivate shard sizing.
- Chapter 7 uses the dispatch pressure to explain why soft-routed MoE is the
  practical public implementation today.
- Chapter 8 uses the host-round-trip cost to justify the cache-friendly Swift
  runtime design.

The book remains a public CoreML porting guide. The adventurous part is the
black-box systems work needed to discover the constraints; the runnable path is
the supported CoreML path.