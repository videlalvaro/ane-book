# ane-models

**Production LLM inference on the Apple Neural Engine** — a practitioner's guide,
complete with converters, Swift runtimes, and validated model manifests.

Every model in this repo runs **100% on the Neural Engine** (verified with
`MLComputePlan`). No GPU fallback. No CPU matmuls.

---

## Models

| Model | Type | Params | ANE tok/s | Status |
|-------|------|--------|-----------|--------|
| [Phi-4-mini-instruct](models/phi4-mini/) | Dense LLM | 3.8B | ~17 | ✅ v1.0 |
| [Hy-MT 1.5](models/hymt/) | Dense translation | 1.8B | ~34 | ✅ v1.0 |
| [ZAYA1-8B](models/zaya/) | MoE LLM | 8B | ~9 | ✅ v1.0 |
| [Privacy Filter](models/privacy-filter/) | MoE NER / PII | ~1.5B | ~24.6 sent/s | ✅ v1.0 |

Hardware: Apple M4 Max, 48 GB unified memory, macOS 15, Xcode 16.

---

## Quick Start

### Prerequisites
- macOS 15+ (Sequoia), Apple Silicon
- Xcode 16+ (for `xcrun coremlcompiler` and coremltools 9)
- Python 3.11+ via Xcode tools

### Run the Privacy Filter demo

```bash
# Build once (downloads weights from HuggingFace, ~3 GB)
/usr/bin/python3 models/privacy-filter/build_scripts/build_pf_packed_alllayers.py

# Extract Swift weights
/usr/bin/python3 models/privacy-filter/build_scripts/extract_pf_swift_weights.py

# Redact a file
bash demo/demo_redact.sh demo/pii_examples.txt
```

### Convert Phi-4-mini from GGUF

```bash
# Download GGUF (requires HuggingFace account for Phi-4)
# Place at: models/phi4-mini/Phi-4-mini-instruct.Q8_0.gguf

# Convert all shards (Xcode python3 only)
/usr/bin/python3 converters/phi4_mini_rangedim_export_shard.py --all

# Convert LM head shards
/usr/bin/python3 converters/phi4_mini_lm_head_shards.py

# Check ANE residency
/usr/bin/python3 validators/phi4_mini_residency_check.py
```

---

## The Apple Neural Engine Inference Book

The Apple Neural Engine Inference Book in `book/` is a chapter-by-chapter porting guide for
practitioners who want to port their own models to ANE:

| Chapter | Topic |
|---------|-------|
| [00 — Modern Inference](book/00-why-ane.md) | Tokens, prefill/decode, KV cache, ANE vs GPU vs CPU, the Conv2d trick |
| [01 — ANE Laws](book/01-ane-laws.md) | Empirical rules: shard limits, quantization, residency |
| [02 — Porting Recipe](book/02-porting-recipe.md) | GGUF → CoreML, step by step |
| [03 — Quantization](book/03-quantization.md) | INT8 production, INT4 tradeoffs, the silent CPU fallback |
| [04 — Shard Sizing](book/04-shard-sizing.md) | Layer count vs size, 250 MB limit, LM-head splits |
| [05 — Stateful KV Cache](book/05-stateful-kv-cache.md) | MLState, Swift daemon design, decode loop |
| [06 — RangeDim + Speculative](book/06-rangedim-speculative.md) | Variable T, n-gram acceptance |
| [07 — MoE on ANE](book/07-moe-on-ane.md) | Soft routing, per-expert dispatch, ZAYA & Privacy Filter |
| [08 — Swift Runtime](book/08-swift-runtime.md) | Cache-friendly CoreML orchestration, state, buffers, and serving |
| [09 — Experiment Index](book/08-experiments.md) | Searchable index of experiment writeups |
| [10 — Decision Journal](book/09-journal.md) | The thinking behind the hard calls |
| [Glossary](book/glossary.md) | Definitions for inference, CoreML, ANE, and validation terms |

---

## Repository Structure

```
ane-models/
├── book/           ← the porting guide (chapters 00–10)
├── converters/     ← Python scripts for GGUF → CoreML (Xcode python3)
├── runtime/        ← Swift inference runtimes
├── models/         ← per-model manifests, goldens, build scripts
│   ├── phi4-mini/
│   ├── hymt/
│   ├── zaya/
│   └── privacy-filter/build_scripts/
├── validators/     ← residency checks + quality gates
├── demo/           ← end-user demos
├── research/       ← findings, negative results, ANE internals
└── blogposts/      ← published and draft blog posts
```

---

## Key Invariants

1. **ANE-only**: every matmul, norm, and activation runs on the Neural Engine.
   `MLComputePlan` must show 100% `ios18.conv` ops on ANE before any benchmark.

2. **Quality before perf**: cosine similarity ≥ 0.97 vs FP16 golden before
   any benchmarking or model shipping.

3. **INT8 per-tensor is the production baseline**. INT4 per-block silently falls
   to CPU on small shards — see [research/INT4_SHARD_ANE_BUG.md](research/INT4_SHARD_ANE_BUG.md).

4. **Shard size ≤ 250 MB**. Above this, ANEF compiler emits error -14.

---

## Research

`research/` contains findings that don't fit in the how-to chapters:

- [ANE_CHAIN_SCHEMA.md](research/ANE_CHAIN_SCHEMA.md) — ObjC runtime reflection of the ANE private API
- [ANE_SCALING_FINDINGS.md](research/ANE_SCALING_FINDINGS.md) — 0.5B → 3B scaling limits
- [INT4_SHARD_ANE_BUG.md](research/INT4_SHARD_ANE_BUG.md) — The silent CPU fallback with INT4 per-block

---

## License

See [LICENSE](LICENSE).
