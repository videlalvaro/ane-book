---
layout: default
title: "Experiment 19 - Balanced Tree Reduction (Semigroup Accumulator)"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="18-constraint-propagation-for-realness.html">Previous: Experiment 18</a> | <a href="20-weighted-automaton-layer-partition-search.html">Next: Experiment 20</a></nav>

# Experiment 19 - Balanced Tree Reduction (Semigroup Accumulator)

**Source**: Stepanov & McJones, *Elements of Programming* — Ch. on associative 
operations and semigroups

**Mathematical basis**:
The EML accumulation `ln(exp(a) + exp(b))` is associative (it's addition in log-space, 
i.e., log-sum-exp defines a semigroup). Currently accumulated linearly (depth K, 
zero ILP). A balanced tree of width W has depth log_W(K) and W-1 independent 
pairs at each level.

Linear (current):
```
acc = op(acc, x[0])  // serial chain, depth K
acc = op(acc, x[1])
...
```

Tree (width 8, proposed):
```
t0 = op(x[0], x[1])  // 4 independent pairs → ILP
t1 = op(x[2], x[3])
t2 = op(x[4], x[5])
t3 = op(x[6], x[7])
u0 = op(t0, t1)       // 2 independent pairs
u1 = op(t2, t3)
result = op(u0, u1)   // final merge
```

**Provenance**:
- Stepanov's key theorem: for any associative binary operation, the 
  number of operations is fixed but the *depth* (critical path length) can be 
  reduced from n to ceil(log2(n)) via balanced tree evaluation.
- This is distinct from Exp 13 (8-wide linear unroll, which failed from register 
  pressure). Tree reduction changes the *dependency structure*, not just the width.
- The semigroup concept ensures correctness: associativity guarantees any 
  parenthesization gives the same result.

**Expected impact**: Better ILP by reducing dependency chain depth. Different 
failure mode than Exp 13 — register pressure is similar but dependency chains 
are logarithmic instead of linear.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="18-constraint-propagation-for-realness.html">Previous: Experiment 18</a> | <a href="20-weighted-automaton-layer-partition-search.html">Next: Experiment 20</a></nav>
