# Factorization Cost Dominates Routing Gain in Sparse Expert Layers

## Abstract

Sparse mixture-of-experts layers are conventionally evaluated against a dense
feedforward baseline. That comparison conflates two distinct effects: the cost of
partitioning a wide feedforward layer into narrow experts, and the benefit of
routing among them. We separate the two with an *all-experts-on* control — the
identical factorized layer with routing disabled — and sweep the number of
experts while holding total hidden width and parameter count fixed. On two
synthetic sequence-to-sequence tasks we find that the factorization penalty grows
with expert count and saturates near 10 points of token accuracy, whereas routing
recovers a roughly constant ~4 points regardless of expert count. Net performance
relative to dense therefore degrades monotonically with expert count, and the
sparse layer is competitive with dense only at very small expert counts. We
additionally report a cautionary case study: omitting this control produced four
successive incorrect conclusions about a routing mechanism across four
architecture generations of our own work.

## 1. Introduction

A sparse expert layer replaces one wide feedforward network (FFN) with `n` narrow
expert FFNs plus a mechanism that activates a subset per input. Reported results
are typically framed as a comparison against a dense FFN at matched parameter
count, and differences are attributed to routing.

That attribution does not follow. Replacing a `d_model → d_ffn → d_model` FFN with
`n` experts of width `d_ffn/n` changes the function class even before any routing
occurs. The second layer becomes **block-diagonal**: each expert's hidden units
project only through that expert's own output matrix, so hidden units belonging to
different experts can never mix. A dense FFN of the same total width has a full
output projection and no such restriction. Any comparison of a routed sparse layer
against dense measures

```
(routed sparse) − dense  =  factorization penalty + routing effect
```

and reports the sum as though it were the second term.

We measure the two terms separately. The instrument is a control we call
**all-experts-on**: the same factorized layer, every expert active, uniform
weights, routing switched off. Then

```
factorization penalty  =  all-experts-on − dense
routing effect         =  routed − all-experts-on
```

Sweeping `n` at fixed total width turns each into a curve.

Our contributions:

1. A controlled decomposition separating factorization cost from routing gain,
   requiring only one additional control condition.
2. Measured curves for both terms across `n ∈ {2, 4, 8, 16, 32}` on two tasks,
   showing the penalty grows and saturates while the gain stays flat.
3. A documented case in which omitting the control yielded four consecutive
   incorrect conclusions, offered as evidence that the control is not optional.

## 2. Setup

**Architecture.** A transformer encoder with learned token and positional
embeddings, `L` layers of multi-head self-attention followed by a variant FFN
block, and a pooled output head predicting a fixed-length output string. Only the
FFN block differs between conditions; embeddings, attention, and head are
identical throughout.

**Dense.** Standard `d_model → d_ffn → d_model` FFN with GELU.

**Factorized (sparse) layer.** `n` experts, each `d_model → d_ffn/n → d_model`,
so total hidden width equals the dense `d_ffn` at every `n`. Expert outputs are
combined by a weight vector `a` normalized to sum to one. Expert weights are
stored batched, so the expert that computes is the expert that is selected.

**Selection.** An expert's drive is the energy of its own first-layer response,
`‖GELU(W₁x + b₁)‖`, divisively normalized across experts so the mean drive is 1.
No parameter is trained to predict relevance before computing. Activation is
`a = B · softmax((drive − 1 + Ca)/τ)` iterated for a small number of steps, where
`C` is a learned expert-to-expert coupling matrix with zero diagonal. The `softmax`
scaling pins total activation mass to `B` by construction, so an expert can enter
the active set only by displacing another. We set `B = n/4`, holding the active
*fraction* at 25% so only granularity varies across the sweep.

**All-experts-on control.** Identical layer and parameter count, with `a = 1` for
every expert (uniform weights, total mass `n`). Selection is fully disabled; we
verify the activation weight standard deviation is exactly zero.

**Metric.** Token accuracy — the fraction of non-padding output positions
predicted correctly — is primary. Exact-match on the full output string is
reported as secondary; it is a brittle measure that reports a model having learned
69% of a task's tokens as 0.9% exact, and it obscures partial learning at the low
end.

### 2.1 Tasks

**Latent-rule.** Input is a five-digit string. One of four algorithms is applied:
reverse, sort ascending, increment each digit mod 10, or running prefix sum mod
10. These require different computational patterns — positional remapping, global
comparison, a local map, and a sequential scan. Crucially, *which* algorithm
applies is not indicated in the input: the selector is `(Σ digits) mod 4`, which
is nonlinear in the input and therefore not readable by a linear gate. The four
rules are exactly balanced and inputs are deduplicated across splits.

We validated the task before use. Trained on a single algorithm with no selector,
the dense model reaches 1.000 token accuracy on reverse and 0.9999 on sort;
prefix sum, the hardest, reaches 0.694. On the full task with the hidden selector
it reaches 0.571. The achievable bar is therefore ~0.92 and the dense baseline sits
at 0.571 — substantial headroom, neither a floor nor a ceiling.

**Arithmetic.** Character-level evaluation of integer arithmetic expressions with
operator precedence and parentheses, 2–3 operations, results in `[-999, 999]`.
Included as an independent second task with different structure: unlike
latent-rule it has no separable subtasks, since evaluating any expression requires
the same parse-and-apply circuit.

## 3. Results

*[RESULTS TABLES — filled from nodes_sweep_3seed.log and the arithmetic sweep]*

### 3.1 The factorization penalty grows and saturates

*[curve table: n vs tax, 3 seeds, both tasks]*

### 3.2 The routing gain does not depend on granularity

*[curve table: n vs gain, 3 seeds, both tasks]*

### 3.3 Net effect degrades monotonically with expert count

*[net table]*

## 4. Case study: what the confound costs

We did not set out to measure this. The decomposition emerged from diagnosing a
sequence of failures in our own architecture work, and the sequence is
instructive.

Over four architecture generations we tested variants of a mechanism in which an
activated expert recruits additional experts through a learned graph: recruitment
by proximity in a learned position space; a complementarity penalty intended to
force recruited experts to compute different functions; a free learned edge
matrix with an independent weight per expert pair; and a resonance-based variant
with unconstrained activation mass. Every variant scored below the dense
baseline. We concluded, repeatedly and in writing, that recruitment harms
accuracy.

Every one of those comparisons used a factorized layer whose total hidden width
was 25% smaller than the dense baseline's, and none included an all-experts-on
control. When we matched the width and added the control, the factorization
penalty at `n = 16` was larger than the entire deficit we had attributed to
recruitment. Measured against a fair within-architecture reference, selection
*helped*.

The failure mode is worth naming precisely. Each generation produced a negative
result; each negative result motivated a new mechanism aimed at the wrong cause;
and because every generation shared the same uncontrolled comparison, no amount
of iteration could have surfaced the error. Four mechanisms agreeing on a null is
easy to read as evidence about mechanisms. It was evidence about the measurement.

## 5. What we do not show

- **Two synthetic tasks, small models.** `d_model = 128`, 2 layers, ~700K
  parameters, 20K training examples. Nothing here establishes behavior at scale
  or on natural data, and the saturation point of the penalty curve may well be
  scale-dependent.
- **One total width.** The sweep varies `n` at fixed `d_ffn = 1024`. Whether the
  penalty curve's shape is invariant to total width is untested.
- **The crossover location is not resolved.** The routed condition is
  substantially noisier across seeds than dense or all-experts-on (spreads of
  ~4 points versus ~0.5 and ~2). Each seed produced some small `n` at which the
  routed layer matched or beat dense, but not consistently the same `n`. We can
  support "the crossover is at small `n`"; we cannot support a specific value.
- **No claim that the routing mechanism is preferable.** Standard top-k routing
  with a comparable gate outperformed our selection mechanism on the latent-rule
  task. This paper is a measurement, not an architecture proposal.
- **Expert specialization did not occur.** We measured whether the active expert
  set identifies which of the four algorithms an input requires, by
  nearest-centroid classification on the activation vectors. Every condition
  scored at chance (0.234–0.256 against a 0.250 floor), including both top-k
  baselines. Whatever the routing gain reflects, it is not subtask
  specialization. This is a negative result for a common motivation for sparse
  expert layers, on a task deliberately built to have separable subtasks.

## 6. Conclusion

The dense-versus-sparse comparison that motivates much sparse-expert work
measures the sum of two effects with opposite signs and different dependence on
expert count. Separating them requires one extra control condition and changes
the conclusion: on our tasks the cost of factorizing a feedforward layer grows
with the number of experts and saturates near 10 points, while routing returns a
roughly constant ~4 points. Sparse expert layers are net-competitive only at
expert counts far below those typically deployed, and the binding constraint is
the partition, not the router.

We recommend reporting the all-experts-on control alongside any dense comparison.
It costs one training run and, in our own case, would have prevented four
successive wrong conclusions.
