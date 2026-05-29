---
layout: default
title: "Chapter 8 - Swift Runtime"
---

# Chapter 8 - Swift Runtime

A converted model is not useful until something can call it one token at a time.
The Swift runtime is that bridge: it loads compiled CoreML shards, owns the
sequence state, feeds each shard the tensors it expects, runs the LM head, chooses
the next token, and keeps the loop warm enough that ANE time is not buried under
host overhead.

The runtime is deliberately plain Swift plus CoreML. The hot path is shaped by a
few constraints:

1. Load models once, not per token.
2. Allocate buffers once, not per token.
3. Keep recurrent state in `MLState` or stable `MLMultiArray` storage.
4. Use direct pointer copies for embeddings, masks, routing weights, and logits.
5. Measure host overhead separately from ANE shard time.

The concrete implementations live in `runtime/phi4_mini_ane.swift`,
`runtime/lfm25_ane.swift`, `runtime/hymt_ane.swift`, and `runtime/zaya_ane.swift`.
They differ by model architecture, but they share the same runtime discipline.

## What the Runtime Owns

The converter produces artifacts. The runtime turns them into an inference loop.

For a dense model such as Phi-4-mini, the runtime owns:

- the runtime manifest (`phi4mini_runtime_meta.json`), which tells Swift which
  shard files exist and what dimensions they use;
- one `MLModel` per layer shard;
- one `MLState` per stateful layer shard;
- LM-head shard models;
- the host-side embedding table;
- reusable input/output buffers for hidden states, RoPE tables, masks, and logits.

For hybrid or MoE models such as LFM2.5 and ZAYA, it also owns routing buffers,
expert-bias tables, conv states, and expert shard dispatch.

The runtime is not a second model implementation. Its job is orchestration. Heavy
projection work belongs inside the compiled CoreML shards; the host should only
do bookkeeping that is cheap, deterministic, and hard to express as an ANE graph.

## Boot Sequence

Startup is intentionally expensive and decode is intentionally cheap.

The runtime does this work once:

```swift
let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
let meta = try JSONDecoder().decode(PhiRuntimeMeta.self, from: metaData)

let cfg = MLModelConfiguration()
cfg.computeUnits = .all

var layerModels = [MLModel]()
var layerStates = [MLState]()
for spec in sortedLayers {
    let path = resolvePath(spec.path, relativeTo: metaPath)
    let model = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
    layerModels.append(model)
    layerStates.append(model.makeState())
}
```

The important detail is `makeState()`: state is created once per loaded stateful
model, then reused across decode calls for one sequence. Creating model objects or
state objects inside the token loop would dominate latency and destroy cache
locality.

Manifest validation also happens before generation starts. The Phi runtime checks
that layer shards cover the full layer range with no gaps or overlaps, and that
LM-head shards start at vocabulary offset 0. That catches broken artifact layouts
before a half-loaded model reaches the hot path.

## The Decode Hot Path

At decode time, the loop is simple:

1. Copy the current token embedding into the reusable hidden-state input buffer.
2. Fill the position-dependent inputs: RoPE, attention mask, write mask, or
   routing weights.
3. Call each layer shard in order, passing the same state objects.
4. Run LM-head shards.
5. Reduce logits on the host and choose the next token.
6. Append the token and repeat.

In pseudocode:

```swift
var hidden = embedToken(currentToken)

for (model, state) in zip(layerModels, layerStates) {
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "hidden": hidden,
        "pos": posArray,
        "rope_cos": ropeCosArray,
        "rope_sin": ropeSinArray,
        "attn_mask": maskArray,
    ])
    let out = try model.prediction(from: input, using: state)
    hidden = out.featureValue(for: "out_hidden")!.multiArrayValue!
}

let next = try runLMHeadAndArgmax(hidden)
```

That loop is only fast if all transient work around it is controlled. The ANE can
run the shard quickly, but Swift can still lose time to heap allocation,
`NSNumber` boxing, repeated trigonometry, and avoidable tensor copies.

## Cache-Friendly Host Design

The LFM2.5 runtime shows the pattern clearly. It allocates all decode-time arrays
once:

```swift
embedBuf     = try MLMultiArray(shape: [1, H,   1, 1] as [NSNumber], dataType: .float32)
writeMaskBuf = try MLMultiArray(shape: [1, 1, SEQ, 1] as [NSNumber], dataType: .float32)
attnMaskBuf  = try MLMultiArray(shape: [1, 1,   1, SEQ] as [NSNumber], dataType: .float32)
ropeCosBuf   = try MLMultiArray(shape: [1, dh,  1, 1] as [NSNumber], dataType: .float32)
ropeSinBuf   = try MLMultiArray(shape: [1, dh,  1, 1] as [NSNumber], dataType: .float32)
routingBuf   = try MLMultiArray(shape: [1, N,   1, 1] as [NSNumber], dataType: .float32)
```

The comments in `runtime/lfm25_ane.swift` capture why this matters: the arrays
are small, but allocating them every token creates heap churn and large numbers of
boxed scalar writes. Reusing them keeps the host side predictable.

Embedding lookup is also a direct memory copy into an existing buffer:

```swift
let emb = embeddings[tokenId]
let dst = embedBuf.dataPointer.assumingMemoryBound(to: Float.self)
emb.withUnsafeBytes { src in
    memcpy(dst, src.baseAddress!, hiddenSize * 4)
}
```

This is the right shape of host work: one contiguous copy, no per-channel object
creation, no temporary tensor allocation. Similar pointer-based writes fill RoPE
buffers, routing buffers, and LM-head reduction buffers.

## Precomputed Tables

RoPE is position-dependent, but its sine and cosine values do not need to be
recomputed from scratch every token. The runtime precomputes tables at startup:

```swift
for pos in 0..<maxSeqLen {
    for i in 0..<halfDh {
        let freq = 1.0 / pow(ropeTheta, Float(2 * i) / Float(dh))
        let angle = Float(pos) * freq
        ropeCosTable[pos * dh + i] = Foundation.cos(angle)
        ropeSinTable[pos * dh + i] = Foundation.sin(angle)
    }
}
```

During decode, the runtime copies the row for the current position into the
CoreML input buffer. This trades a little startup memory for a stable per-token
cost and removes trigonometric calls from the hot path.

Masks follow the same principle. A full attention mask starts in a known state,
then each step mutates only the positions that changed. For LFM2.5, `vDSP_vfill`
resets the mask to `-1e4` quickly; the write mask is zeroed and a single current
position is set.

## State Ownership

State is where the runtime can accidentally become slow or wrong.

For standard transformer attention, state means KV cache. The preferred CoreML
shape is `MLState`: each layer shard owns persistent K/V tensors, and Swift passes
the state object into `prediction(from:using:)`. This avoids re-sending the whole
prefix cache as normal inputs every token.

For models with short convolution state, such as LFM2.5, the state is tiny and
fixed-size: a sliding window of three positions per ShortConv layer. The runtime
keeps those arrays stable and swaps in the updated output after each shard call.

The design rule is the same in both cases: state should be allocated once at
sequence start, mutated in place or replaced by shard outputs, and reset only when
a new sequence begins.

## LM Head and Sampling

The LM head is often too large for one CoreML package, so the runtime loads
multiple LM-head shards. Each shard produces a slice of vocabulary logits. The
host then finds the best token across slices.

This host reduction is acceptable because it is linear over logits and small
compared with running the transformer stack. The expensive part, the projection
from hidden state to logits, remains in the compiled CoreML shards.

The runtime should avoid constructing a full temporary vocabulary array unless a
sampler needs it. Greedy decode can reduce shard outputs directly to `(token,
score)` pairs. More advanced sampling can still be implemented, but it should be
profiled separately because top-k, temperature, and grammar constraints can move
work back onto the host.

## Serving Mode

The Phi runtime includes a serve mode because process lifetime matters. Loading
CoreML shards, creating states, and paying first-use specialization costs are not
per-request work in production. A daemon process should load the model once,
accept requests, reset sequence state, generate, and stay warm.

The serving boundary also gives a clean place to expose timings:

- prefill tokens and seconds;
- decode tokens and seconds;
- layer forward time;
- embedding, mask, LM-head copy, prediction, and reduction time.

Those counters are not decoration. They tell you whether the bottleneck is ANE
execution, LM-head reduction, state setup, or host tensor plumbing.

## Failure Modes

The runtime can make a correct model look slow. Common mistakes:

- creating `MLModel`, `MLState`, `MLDictionaryFeatureProvider`, or
  `MLMultiArray` objects inside avoidable inner loops;
- filling tensors through subscripted `NSNumber` access instead of direct
  pointers;
- recomputing RoPE tables or full masks every token;
- copying full KV caches through the host instead of using `MLState`;
- running LM-head shards serially when shard outputs could be reduced more
  carefully;
- benchmarking cold model load and warm decode as one number.

The fix is not to guess. Profile the host buckets, isolate the ANE shard time,
then remove the largest host-side source of per-token work.

## Checklist

```text
[ ] Runtime manifest validates layer coverage and shard paths before generation
[ ] MLModel objects are loaded once per process
[ ] MLState objects are created once per sequence, not once per token
[ ] Decode-time MLMultiArrays are pre-allocated and reused
[ ] Embedding, masks, RoPE, routing, and logits use direct pointer access
[ ] RoPE tables are precomputed for max_seq_len
[ ] KV cache or conv state is reset only at sequence boundaries
[ ] LM-head shard reduction is measured separately from layer execution
[ ] Benchmarks separate cold load, prefill, warm decode, and host overhead
```
