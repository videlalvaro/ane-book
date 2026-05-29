---
layout: default
title: "Experiment 18 - Constraint Propagation for Realness"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="17-fused-apl-style-inner-product.html">Previous: Experiment 17</a> | <a href="19-balanced-tree-reduction-semigroup-accumulator.html">Next: Experiment 19</a></nav>

# Experiment 18 - Constraint Propagation for Realness

**Sources**: Dechter's *Constraint Processing* (Arc Consistency, AC-3) + TAOCP Vol. 4 
Fascicle 7 (Constraint Satisfaction)

**Formulation as CSP**:
- Variables: each node in the EML computation graph
- Domain: {real, complex}
- Constraints:
  - "final output must be real"
  - "ln(positive_real) → real"
  - "exp(real) → positive_real"
  - "real + real → real"
  - "positive_real × positive_real → positive_real"

Run AC-3 backwards from outputs. Any node proven to be in the "real" domain 
uses f64 ops instead of Complex64.

**Provenance**:
- The CSP formulation maps directly to Dechter's framework: variables = graph nodes, 
  domains = {real, complex}, constraints = type rules.
- AC-3 (arc consistency algorithm 3) from Dechter Ch. 3 is the workhorse: 
  iterate until fixpoint, propagating domain reductions.
- TAOCP Fascicle 7's treatment of constraint satisfaction provides the 
  backtracking framework for cases where AC-3 alone is insufficient.
- This generalizes our best single optimization (Exp 6, real-exp bypass, ~40% speedup) 
  from hand-coded matmul-only to *all* operations (softmax, RMSNorm, SiLU, RoPE).

**Expected impact**: Generalize real-bypass to all ops. Could be significant for 
softmax and RMSNorm which also have known-real intermediate values.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="17-fused-apl-style-inner-product.html">Previous: Experiment 17</a> | <a href="19-balanced-tree-reduction-semigroup-accumulator.html">Next: Experiment 19</a></nav>
