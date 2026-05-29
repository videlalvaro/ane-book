---
layout: default
title: "Journal 063 - Stream-Level Two-Model Chain Breakthrough"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="062-stream-setupoperationforinputfeatures-replaces-pool.html">Previous: Journal 062</a> | <a href="064-phi-stateful-stream-chain-smoke.html">Next: Journal 064</a></nav>

# 2026-04-28 - Stream-Level Two-Model Chain Breakthrough

**Intent**: Prove true cross-model chaining inside one stream-level execution stream by using stream-level encode hooks instead of public host roundtrips, following the validation-first notes measurement/validation discipline and the [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md) focus on stream-level ANE behavior.

**Setup**: CoreML imports stream-level symbols from CoreML execution runtime, including stream-level prepare and encode hooks. `local two-stage stream probe` now resolves those runtime symbols. Successful sequence: bind stage A input/output via CoreML operation methods; bind stage A output feature/memory object into stage B input binder; bind stage B output; call stream-level prepare+encode for stage A and stage B operation handles into one stream; then call the CoreML stream execution path.

**Result**: PASS on the tiny distinct-input control `toy_a(x+1) -> toy_b_h(h*2)` with no `h` provider. Stage A hidden was `[2, 3, 4, 5]`; stage B hidden was `[4, 6, 8, 10]`. This proves stage B consumed stage A's output buffer and is the first true two-model stream-level chain in this repo.

**Surprise / hurdle**: The key unlock was discovering that the public CoreML/CoreML execution runtime stack already imports the stream-level prepare/encode symbols, making it possible to encode two operation handles into one stream while wiring the intermediate through runtime binders.

**Lesson**: Stream-Level encode access turns cross-model chaining from a host-roundtrip problem into a stream construction problem.

**Next**: Validate the same path on two Phi layer-range shards against the public host-roundtrip output, then profile hidden-state copy removal.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="062-stream-setupoperationforinputfeatures-replaces-pool.html">Previous: Journal 062</a> | <a href="064-phi-stateful-stream-chain-smoke.html">Next: Journal 064</a></nav>
