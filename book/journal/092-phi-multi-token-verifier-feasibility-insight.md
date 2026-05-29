---
layout: default
title: "Journal 092 - Phi Multi-Token Verifier Feasibility Insight"
---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="091-phi-n-gram-force-mode-tried.html">Previous: Journal 091</a> | <a href="093-phi-t-4-verifier-op-pattern-probe-passed.html">Next: Journal 093</a></nav>

# 2026-04-29 - Phi Multi-Token Verifier Feasibility Insight

**Intent**: Refine the public Phi-4-mini speculative verifier design after n-gram simulations showed a draft-4 pass-count target near 2x, using Knuth string matching and Concrete Mathematics amortization framing from the validation-first notes plus speculative decoding (Leviathan et al., 2023).

**Setup**: Design note only; no command run. Target implementation is a public CoreML stateful block layer shard with `T=4`: `x [1,d,T,1]`, `rope [T,d_half]`, `attn_mask [1,1,T,max_seq]`, `kv_write_mask [1,1,max_seq,T]`, and output hidden `[1,d,T,1]`, paired with the existing batch-4 LM-head shards.

**Result**: New insight: exact verification likely does not require cheap full-`MLState` rollback if the block verifier writes draft KVs into future positions. Unaccepted future KV slots remain hidden by `attn_mask`, and the first rejected slot is overwritten when the target fallback token is processed at that same position. Public CoreML state access exists, but copying full KV is not the desired speed path.

**Surprise / hurdle**: The commit/discard mechanism may be expressible through future-position writes plus masks, shifting the hard requirement from state rollback to building a correct ANE-resident `T=4` stateful layer artifact.

**Lesson**: For public exact speculation, avoid host KV copies; make speculative state cheap by writing only into masked future slots and committing positions by advancing the attention mask.

**Next**: Build the smallest representative `T=4` block layer shard and run strict ANE residency before scale; then run golden equivalence against four single-token steps and connect it to the already gated batch-4 LM head only if residency and quality pass.

**Refs**: [research/ANE_CHAIN_SCHEMA.md](https://github.com/videlalvaro/ane-book/blob/main/research/ANE_CHAIN_SCHEMA.md)

---

<nav class="experiment-nav"><a href="../09-journal.html">Back to Journal Index</a> | <a href="091-phi-n-gram-force-mode-tried.html">Previous: Journal 091</a> | <a href="093-phi-t-4-verifier-op-pattern-probe-passed.html">Next: Journal 093</a></nav>
