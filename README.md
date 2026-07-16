# Dynamic Coalition Network (DCN)

A neural architecture where compute nodes live in a learned embedding space and self-organize into task-specific coalitions through spatial proximity. Instead of fixed layers or static expert routing, nodes recruit neighbors based on where they are in the embedding space — nodes that handle similar computations cluster together through gradient descent alone.

## How It Works

The DCN replaces the standard feedforward layer in a transformer with a **Coalition-Graph FFN**:

1. **Seed Selection** — For each input, the network picks the top-k most relevant nodes based on learned key vectors
2. **Proximity Recruitment** — Active seeds recruit nearby nodes in the embedding space using cosine similarity
3. **Coalition Compute** — Only the activated coalition processes the input, with contributions weighted by activation strength

Nodes have learned positions in a 32-dimensional space. During training, nodes that frequently co-activate on similar tasks get pulled closer together. The result is an emergent graph structure where clusters of nodes correspond to different computational roles.

## Architecture

```
Input → Token Embedding → Positional Encoding
      → [Attention → LayerNorm → Coalition-Graph FFN → LayerNorm] × 4
      → Output Head → Prediction
```

Each Coalition-Graph FFN contains:
- **16 compute nodes** (small feedforward networks)
- **16 key vectors** for input-dependent seed selection
- **16 position vectors** in 32-dim embedding space
- **Learned recruitment threshold**

Temperature annealing (cosine schedule 1.0 → 0.1) sharpens routing decisions over training.

## Task

Synthetic compositional arithmetic — the network learns to evaluate expressions like `6*(8+9+7)` or `5/3*3+9`. Expressions are balanced across four types:

| Type | Example |
|------|---------|
| Addition/Subtraction | `4+3-7` |
| Multiplication/Division | `3*9/3` |
| Mixed Precedence | `5+3*2` |
| Parenthesized | `6*(8+9)` |

The task tests whether different node coalitions emerge for structurally different computations.

## Project Structure

```
├── src/
│   ├── data/arithmetic.py        # Dataset generation + tokenization
│   ├── models/
│   │   ├── shared.py             # Base transformer (embedding, attention, output head)
│   │   ├── coalition.py          # Coalition-Graph FFN + CoalitionModel
│   │   ├── dense.py              # Dense baseline
│   │   └── moe.py                # Mixture-of-Experts baseline
│   ├── training/losses.py        # Load balance, size regularization, recruitment losses
│   └── evaluation/
│       ├── metrics.py            # Accuracy, activation pattern collection
│       └── visualize.py          # Heatmaps, position plots, graph structure
├── configs/experiment.yaml       # All hyperparameters
├── train_coalition.py            # Training script with coalition diagnostics
└── notebooks/run_experiment.ipynb # Full experiment pipeline (Colab-ready)
```

## Running

**Colab (recommended):**

Open `notebooks/run_experiment.ipynb` and run all cells. Trains on A100/T4 GPU in ~1 hour.

**Local:**

```bash
pip install -r requirements.txt
python train_coalition.py
```

Add `--small` for a quick 20-epoch test run on reduced data.

## What Gets Measured

- **Activation heatmaps** — which nodes fire for which expression types
- **Node position clustering** — do nodes self-organize into spatial groups
- **Cluster purity** — do activation patterns map cleanly to operation types
- **Coalition size distribution** — do harder inputs recruit more nodes
- **Per-node type preference** — does each node specialize

## Config

Key parameters in `configs/experiment.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Model dimension |
| `n_layers` | 4 | Transformer layers |
| `n_nodes` | 16 | Compute nodes per layer |
| `n_seeds` | 4 | Seeds selected per input |
| `d_pos` | 32 | Position embedding dimension |
| `target_coalition_size` | 7 | Target active nodes per input |
| `epochs` | 50 | Training epochs |
