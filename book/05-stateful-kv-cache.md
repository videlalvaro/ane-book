# Chapter 5 — Stateful KV Cache with MLState

## Why MLState

Without a stateful cache, every decode step must pass the entire KV prefix
(all past keys and values) as CoreML inputs. At seq_len=512 with d_kv=128 and
32 heads, that's 2 × 32 × 512 × 128 = 4M floats per layer, per step.

CoreML's `MLState` stores tensors inside the model's runtime, persisting between
`predict()` calls. The host never touches the KV tensor after write.

The decode loop reduces to:
```
token → embed → [layer_0(state), layer_1(state), ..., layer_N(state)] → lm_head → sample
```

Each `layer_i(state)` call reads past KV from `state`, appends the new KV, writes
back — all in CoreML. The host passes only the current token's hidden state.

---

## MLState in coremltools

### Converting a Stateful Layer

```python
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# State specification
k_state_spec = ct.StateType(
    wrapped_type=ct.TensorType(shape=(1, n_kv_heads, max_seq_len, d_head)),
    name="k_cache",
)
v_state_spec = ct.StateType(
    wrapped_type=ct.TensorType(shape=(1, n_kv_heads, max_seq_len, d_head)),
    name="v_cache",
)

model_stateful = ct.convert(
    traced_layer,
    inputs=[ct.TensorType(name="hidden", shape=[1, d_model, 1, 1])],
    states=[k_state_spec, v_state_spec],
    outputs=[ct.TensorType(name="out_hidden")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS15,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
```

### Writing to State in the MIL Graph

Inside the traced PyTorch model, use CoreML's `coremltools.converters.mil.frontend.torch`
state write op:

```python
# In the traced model's forward pass:
# Write updated keys to state slot 0 (k_cache)
# This is done implicitly by using scatter ops that coremltools recognizes
# as state writes when the input has a StateType shape.
```

The standard pattern is to maintain `k_cache` as a pre-allocated tensor of shape
`[1, n_kv_heads, max_seq_len, d_head]`, use a position counter, and scatter
the new key into position `pos`:

```python
# PyTorch forward (will be traced)
def forward(self, hidden, k_cache, v_cache, pos):
    # ... compute Q, K, V from hidden ...
    # Update cache at position pos
    k_cache[:, :, pos:pos+1, :] = k_new  # CoreML sees this as a state write
    v_cache[:, :, pos:pos+1, :] = v_new
    # Attention using full k_cache[:, :, :pos+1, :]
    ...
```

---

## Swift Runtime for Stateful Decode

The Swift host allocates one `MLState` per active sequence and passes it on
every predict call:

```swift
import CoreML

class ANEDecoder {
    let layers: [MLModel]
    let embeddings: [Float]
    var state: MLState?

    init(layerURLs: [URL], embedURL: URL, config: MLModelConfiguration) throws {
        self.layers = try layerURLs.map { try MLModel(contentsOf: $0, configuration: config) }
        self.embeddings = loadEmbeddings(from: embedURL)
    }

    func beginSequence() throws {
        // Allocate a fresh state for a new sequence
        self.state = try layers[0].makeState()
    }

    func decode(tokenId: Int, pos: Int) throws -> MLMultiArray {
        // 1. Embed
        var hidden = embedToken(tokenId)  // [1, d_model, 1, 1]

        // 2. Run each layer with shared state
        for layer in layers {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLMultiArray(hidden),
                "pos":    MLMultiArray([Int32(pos)]),
            ])
            let out = try layer.prediction(from: input, using: state!)
            hidden = out.featureValue(for: "out_hidden")!.multiArrayValue!
        }

        return hidden  // pass to LM head
    }
}
```

Key points:
- One `MLState` object is shared across all `predict()` calls for a sequence.
- `makeState()` is called once at sequence start, NOT on every token.
- `state` is passed as the second argument to `prediction(from:using:)`.
- After `endSequence()`, the state object can be discarded (ARC releases it).

---

## The CCA (Cross-Chunk Attention) Pattern

For models with chunked prefill (processing T tokens at once, then switching to
decode T=1), the state must be pre-filled with the prefix before decode starts.

```swift
// Prefill: run with T=4 (or however large your RangeDim allows)
for chunkStart in stride(from: 0, to: promptLen, by: T) {
    let chunk = promptTokens[chunkStart ..< min(chunkStart+T, promptLen)]
    try runPrefill(tokens: chunk, startPos: chunkStart)  // writes to state
}
// Decode: run T=1 from promptLen onward
while !done {
    let nextToken = try decode(tokenId: lastToken, pos: currentPos)
    currentPos += 1
}
```

The state accumulates KV pairs during prefill, then decode reads them naturally.
No special handling needed — `MLState` persists across both `prediction()` calls.

---

## Known Bugs and Pitfalls

### The Stateful KV Cache RangeDim Bug (2026-04-xx)

Converting a stateful model with `RangeDim` (variable T) and state writes in the
same graph sometimes produces incorrect state write addresses at T > 1.

**Symptom**: Correct output at T=1, silent corruption at T=2+.

**Mitigation**: Validate with `compare_logits.py` at both T=1 and T=4 before shipping.
If T>1 diverges: revert to fixed T=1 for state shards (use RangeDim only for
non-stateful prefill shards, or split prefill/decode into separate shard sets).

### State Size Must Be Pre-Allocated at Max Seq Len

MLState KV cache tensors must be allocated at `max_seq_len`, not dynamically
grown. Set `max_seq_len` conservatively — 4096 is a good default for most models.

For a 32-layer model with `n_kv_heads=8`, `d_head=128`, `max_seq_len=4096`:
```
KV cache memory = 32 layers × 2 (K+V) × 8 heads × 4096 tokens × 128 dims × 2 bytes (fp16)
               = 32 × 2 × 8 × 4096 × 128 × 2 = ~4 GB
```

On M4 Max (48 GB unified memory) this is fine. On 16 GB machines, reduce
`max_seq_len` or use fewer layers per state bundle.

---

## Checklist

```
[ ] StateType spec declares correct shape [1, n_kv_heads, max_seq_len, d_head]
[ ] makeState() called once per sequence (not per token)
[ ] state passed to every prediction() call
[ ] T=1 and T>1 outputs validated against PyTorch golden
[ ] max_seq_len fits in unified memory budget
```
