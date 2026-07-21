# Dynamic Coalition Network (DCN)

A sparse-routing architecture where compute nodes hold **learned positions** in an
embedding space, and activating one node **recruits its neighbours** into the
coalition that answers a given input.

Built from scratch as an independent exploration. **It turned out to overlap
substantially with published work — see [Relationship to prior work](#relationship-to-prior-work).**
Results below are reported as-is, including the parts that did not work.

---

## The mechanism

Each transformer FFN block is replaced by 16 small expert "nodes". Routing runs
in two stages:

**1. Seed selection** — mean-pool the input, dot against each node's learned key,
take top-k as seeds.
*Standard learned routing. Not novel.*

**2. Proximity recruitment** — each node also carries a learned **position**
vector. Nodes near an active seed get recruited into the coalition, gated by a
learned threshold:

```
proximity = seed_activation @ cosine_sim(positions)
recruit   = sigmoid((proximity - threshold) / tau)
coalition = seeds + recruit
```

In a flat MoE, expert 3 and expert 7 have no relationship to each other. Here
they do — and which nodes end up close together is discovered by gradient
descent, not hand-designed.

Architecture: `Embedding → [Attention → LayerNorm → Coalition-Graph FFN → LayerNorm] × 4 → Output head`,
~3.3M params, d_model 256, 16 nodes/layer, 32-dim positions. Temperature anneals
1.0 → 0.1 to sharpen routing over training.

---

## Task

Character-level compositional arithmetic (`3+5*2-1` → `12`), 80K training
examples, balanced across four expression types: addition/subtraction,
multiplication/division, mixed precedence, and parenthesized. The generalization
split holds out longer and more deeply nested expressions than training.

---

## Results

### Main comparison

| model | params | test acc | gen acc |
|---|---|---|---|
| Dense | 3,263,844 | **0.9324** | 0.0486 |
| DCN (v5) | 3,297,640 | 0.8860 | 0.0550 |
| MoE (top-2/16) | 3,295,588 | *incomplete* | *incomplete* |

**DCN does not beat a dense baseline on this task.** It is also not faster — all
16 nodes are computed and then weighted, so this implementation saves no
wall-clock time. *(The MoE run was cut short by a compute limit.)*

### The ablation that matters

Does proximity recruitment do anything, or is it just an expensive way to
activate more nodes? Four conditions, identical data and init, 60 epochs,
**1 seed**:

| condition | recruit | k | activation mass | test acc |
|---|---|---|---|---|
| `A_orig` (as shipped) | on | 4 | 6.7 | 0.7638 |
| `A_full` (magnitude-controlled) | on | 4 | 6.7 | **0.7600** |
| `B_seedsonly` | off | 4 | 4.0 | 0.7166 |
| `C_matchedk` | off | 7 | 7.0 | **0.7396** |

- vs seeds-only: **+4.3%** — expected; more nodes are active
- vs **matched-k top-k: +2.05%** — recruitment beats plain top-k *while using
  slightly less compute* (6.7 vs 7.0 mass)

`A_full` vs `C_matchedk` is the real test. Beating `B` is trivial, so the control
gives plain top-k the *same activation mass*. Recruitment still wins.

Output magnitude is held at 1 across conditions (`norm_mode='sum'`), because v5
divides by `n_seeds` while activating more than `n_seeds` worth of mass —
without that control, recruitment would "win" simply by being louder.

> ⚠️ **Single seed.** The `±0.0000` in raw output is n=1, not zero variance.
> A 2% gap needs 2–3 seeds before it should be believed. This is the most
> important missing experiment.

### Specialization

Nodes partition by operation type without being told to. Layer 1 cluster purity
**0.749** (one cluster at 1.00):

| node | prefers | strength |
|---|---|---|
| 6, 10 | `mul_div_only` | 1.00 |
| 1 | `add_sub_only` | 1.00 |
| 8 | `parenthesized` | 0.99 |
| 13, 14 | `parenthesized` | 0.92, 0.91 |

Mean coalition size 5.3 / 16, std 1.12 — input-dependent, not fixed.

### What did not work

- **No accuracy win over dense** (0.886 vs 0.932).
- **No compute saving.** All nodes are computed, then masked.
- **Compositional generalization was never actually tested.** The gen split
  scored ~5% for *every* model including dense, so it cannot discriminate
  between them. The original hypothesis — that recruitment improves
  compositional generalization — is **untested**, not disproven.
- **Recruitment reduced per-type distinctness.** Type-signature similarity was
  0.713 for seeds-only vs 0.887 with recruitment: accuracy improved while
  specialization *blurred* — the opposite of the stated hypothesis. The
  mechanism helps, but not for the predicted reason.
- Deeper layers specialize less (purity 0.749 → ~0.48 by layer 4).

---

## Relationship to prior work

Both stages have prior art. This was discovered **after** implementation.

**Stage 1** (keys → top-k seeds) is standard learned routing, as in PEER and
product-key memory. Never claimed as novel.

**Stage 2** overlaps with
**[Modeling Expert Interactions in Sparse Mixture of Experts via Graph Structures](https://arxiv.org/abs/2510.16411)**
("SymphonySMoE", Oct 2025), which builds a graph between experts and co-activates
neighbours. Two implementation differences remain:

| | SymphonySMoE | DCN |
|---|---|---|
| graph construction | expert **weight similarity**, thresholded, static | **learned positions**, gradient descent |
| how the graph is used | **masks** which experts may co-activate | **additively recruits** neighbours |

SymphonySMoE does not compare against a *learned* adjacency. Whether a
gradient-learned expert graph beats a constructed one is an open question — that
is the experiment in `graph_mode_comparison.py`.

Also relevant: [Self-Routing](https://arxiv.org/abs/2604.00421) (parameter-free
routing from hidden states), and
[The Myth of Expert Specialization in MoEs](https://arxiv.org/pdf/2604.09780),
which argues apparent expert specialization reflects geometry rather than domain
expertise — consistent with the specialization decay observed here at depth.

**Honest summary:** independent rediscovery of expert-graph routing, with a clean
ablation isolating the recruitment mechanism, and a different (learned) way of
building the graph.

---

## Repo

```
src/models/coalition.py          DCN — seeds + proximity recruitment
src/models/{dense,moe}.py        baselines
train_coalition.py               main training + diagnostics
recruitment_ablation.py          the 4-condition ablation above
graph_mode_comparison.py         learned vs constructed graph (open question)
pattern_probe.py                 per-node specialization analysis
notebooks/run_experiment.ipynb   full pipeline (Colab)
```

```bash
python train_coalition.py                        # train DCN
python train_coalition.py --small                # quick 20-epoch smoke test
python recruitment_ablation.py --full --seeds 3  # the ablation (use 3 seeds)
```

## Status / next steps

1. **Re-run the ablation with 3 seeds.** Everything hinges on whether +2% survives.
2. Finish the MoE baseline.
3. Learned vs constructed graph — the open question above.
4. Build a generalization split the baselines can actually score on, so the
   original compositional hypothesis becomes testable at all.
