import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter


def plot_training_curves(histories: dict[str, dict], save_path: str = 'training_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, hist in histories.items():
        axes[0].plot(hist['train_loss'], label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, hist in histories.items():
        axes[1].plot(hist['val_loss'], label=name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for name, hist in histories.items():
        axes[2].plot(hist['val_accuracy'], label=name)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_comparison_table(results: dict, save_path: str = 'comparison_table.png'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = ['Model', 'Test Acc', 'Gen Acc', 'Inference (ms)', 'Active Params', 'Specialization']
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r.get('test_accuracy', 0):.1%}",
            f"{r.get('gen_accuracy', 0):.1%}",
            f"{r.get('inference_ms', 0):.2f}",
            f"{r.get('active_params', 0):,}",
            r.get('specialization', 'N/A'),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison table to {save_path}")


def plot_activation_heatmap(patterns: list[dict], layer_idx: int = 0, save_path: str = 'activation_heatmap.png'):
    type_order = ['add_sub_only', 'mul_div_only', 'mixed_precedence', 'parenthesized']
    sorted_patterns = sorted(patterns, key=lambda p: type_order.index(p['expr_type']) if p['expr_type'] in type_order else 99)

    activations = []
    labels = []
    for p in sorted_patterns:
        if layer_idx < len(p['activations']):
            activations.append(p['activations'][layer_idx])
            labels.append(p['expr_type'])

    if not activations:
        print("No activations to plot")
        return

    act_matrix = np.array(activations).T

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(act_matrix, ax=ax, cmap='YlOrRd', vmin=0, vmax=1, cbar_kws={'label': 'Activation Score'})
    ax.set_ylabel('Node Index')
    ax.set_xlabel('Examples (sorted by type)')
    ax.set_title(f'Coalition Node Activations — Layer {layer_idx + 1}')

    type_counts = Counter(labels)
    boundaries = []
    pos = 0
    for t in type_order:
        if t in type_counts:
            mid = pos + type_counts[t] // 2
            boundaries.append((pos, mid, t))
            pos += type_counts[t]

    for start, mid, t in boundaries:
        ax.axvline(x=start, color='white', linewidth=2, alpha=0.8)
        ax.text(mid, -0.5, t, ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved activation heatmap to {save_path}")


def plot_graph_structure(patterns: list[dict], edge_weights: np.ndarray, save_path: str = 'graph_structure.png'):
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed, skipping graph visualization")
        return

    n_nodes = edge_weights.shape[0]
    activations = np.array([p['activations'][0] for p in patterns if len(p['activations']) > 0])

    if len(activations) == 0:
        return

    type_labels = [p['expr_type'] for p in patterns if len(p['activations']) > 0]
    types = list(set(type_labels))
    type_colors = {'add_sub_only': '#2196F3', 'mul_div_only': '#FF9800', 'mixed_precedence': '#4CAF50', 'parenthesized': '#E91E63'}

    node_dominant_type = []
    for node_idx in range(n_nodes):
        type_scores = {t: 0.0 for t in types}
        for i, t in enumerate(type_labels):
            type_scores[t] += activations[i, node_idx]
        dominant = max(type_scores, key=type_scores.get)
        node_dominant_type.append(dominant)

    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)

    threshold = 0.3
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and edge_weights[i, j] > threshold:
                G.add_edge(i, j, weight=edge_weights[i, j])

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2, seed=42)

    node_colors = [type_colors.get(node_dominant_type[i], '#999') for i in range(n_nodes)]
    node_sizes = [activations[:, i].mean() * 2000 + 200 for i in range(n_nodes)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=[w * 3 for w in weights],
                               alpha=0.5, arrows=True, arrowsize=15)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=t)
                       for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', title='Dominant Type')
    ax.set_title('Coalition Graph Structure — Node Specialization & Learned Edges')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved graph structure to {save_path}")


def plot_coalition_size_distribution(patterns: list[dict], layer_idx: int = 0, save_path: str = 'coalition_sizes.png'):
    sizes = []
    for p in patterns:
        if layer_idx < len(p['activations']):
            sizes.append((p['activations'][layer_idx] > 0.5).sum())

    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes, bins=range(0, max(sizes) + 2), edgecolor='black', alpha=0.7, color='#4472C4')
    ax.set_xlabel('Coalition Size (nodes with activation > 0.5)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Coalition Sizes Across Examples')
    ax.axvline(x=np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coalition size distribution to {save_path}")


def compute_cluster_purity(patterns: list[dict], n_clusters: int = 8, layer_idx: int = 0) -> dict:
    activations = []
    labels = []
    for p in patterns:
        if layer_idx < len(p['activations']):
            activations.append(p['activations'][layer_idx])
            labels.append(p['expr_type'])

    if len(activations) < n_clusters:
        return {'purity': 0.0, 'n_clusters': 0}

    X = np.array(activations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)

    total_correct = 0
    cluster_info = {}
    for c in range(n_clusters):
        mask = kmeans.labels_ == c
        cluster_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        if not cluster_labels:
            continue
        counts = Counter(cluster_labels)
        dominant_type, dominant_count = counts.most_common(1)[0]
        total_correct += dominant_count
        cluster_info[c] = {
            'dominant_type': dominant_type,
            'purity': dominant_count / len(cluster_labels),
            'size': len(cluster_labels),
            'distribution': dict(counts),
        }

    overall_purity = total_correct / len(labels)
    return {'purity': overall_purity, 'n_clusters': n_clusters, 'clusters': cluster_info}
