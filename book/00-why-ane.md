# Chapter 0 — Why ANE? The Case for the Neural Engine

## The Hardware You're Not Using

Every Apple Silicon Mac and iPhone ships with an Apple Neural Engine. On an M4 Max
it has 38 TOPS of throughput, runs at a fraction of the power of the GPU, and is
sitting idle while llama.cpp burns CPU and Metal burns battery.

The reason nobody uses it directly is that Apple doesn't document it. There is no
ANE SDK. The only official path is CoreML — and CoreML's documentation is written
for app developers who want to run Vision models, not for researchers who want to
run 8-billion-parameter MoE transformers at 9 tok/s.

This book documents what it takes to get real LLMs running on the ANE, based on
35+ experiments across five model families.

## ANE vs GPU vs CPU — the Real Tradeoffs

| Property | ANE | GPU (Metal) | CPU |
|----------|-----|-------------|-----|
| Peak compute | 38 TOPS (M4 Max) | ~14 TFLOPS fp16 | ~1 TFLOPS |
| Power at load | ~2–4W | ~10–20W | ~8–15W |
| Memory bandwidth | On-chip SRAM + UMA | UMA (shared) | UMA |
| Latency per op | Low (fixed-function) | Higher (shader launch) | High |
| Programmability | CoreML only | Metal Shaders | Anything |
| INT8 native | Yes | Yes (via Metal) | Yes |
| Stateful cache | Yes (MLState) | Manual | Manual |

The ANE wins on power-per-token. At decode speed, the limiting factor is memory
bandwidth (loading weights per token), and the ANE's fixed-function conv engines
are faster per watt at `y = Wx` (the dominant operation in every LLM) than a
general-purpose shader grid.

**The catch**: the ANE only runs what CoreML lets through. And CoreML is picky.

## What CoreML Actually Does

CoreML is a compiler that takes a PyTorch or TorchScript model and emits an
`.mlpackage`. When you call `xcrun coremlcompiler compile`, it produces an
`.mlmodelc` directory with:

- `model.espresso.net` — the network graph as a flatbuffer
- `model.espresso.shape` — tensor shape metadata
- `*.mlmodelc/Data/com.apple.CoreML/weights/` — quantized weight blobs

At runtime, the CoreML framework dispatches ops to one of three backends:
- **ANE** — Apple Neural Engine (fastest for conv/matmul, stateful)
- **GPU** — Metal Performance Shaders
- **CPU** — Accelerate/vDSP

The dispatch decision is made by the **ANE compiler (ANEF)** during `.mlmodelc`
compilation, not at runtime. You can inspect it with `MLComputePlan` —
the ground-truth source of whether your ops land on ANE or not.

## Why Every Matmul Must Be Conv2d

The ANE's primary instruction is a **2D convolution**. The GPU has general
matrix-multiply (GEMM) shaders. The ANE does not have a standalone GEMM.

The trick: a 1×1 convolution over a `[1, C_in, T, 1]` input with a
`[C_out, C_in, 1, 1]` kernel is mathematically identical to `y = Wx` with
`W` being `[C_out, C_in]`.

```python
# This is how every LLM linear projection becomes ANE-native:
self.proj = nn.Conv2d(d_in, d_out, kernel_size=1, bias=False)
# input:  [batch=1, d_in, seq_len, 1]
# output: [batch=1, d_out, seq_len, 1]
```

This is the single most important fact in this book. If you don't reshape
everything to `[1, channels, T, 1]` and use `Conv2d(1×1)`, nothing lands on ANE.

## What This Book Covers

- **Chapter 1** — ANE empirical laws (shard limits, quantization, verified rules)
- **Chapter 2** — Porting recipe (GGUF → CoreML, step by step)
- **Chapter 3** — Quantization (INT8 production, INT4 tradeoffs, the silent CPU fallback)
- **Chapter 4** — Shard sizing (layer count vs size, the 250 MB cliff, LM-head splits)
- **Chapter 5** — Stateful KV cache (MLState, the Swift daemon design)
- **Chapter 6** — RangeDim and speculative decode (T=1..4, n-gram acceptance)
- **Chapter 7** — MoE on ANE (soft routing, per-expert dispatch, ZAYA)
- **Chapter 8** — Experiment log (all 35+ experiments, what worked and why)
- **Chapter 9** — Decision journal (the thinking behind the hard calls)
