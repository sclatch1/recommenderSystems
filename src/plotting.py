"""
All matplotlib-producing functions. Pure data/reporting helpers (stats
tables, text reports, DataFrame-only aggregation) live in src/utils.py
instead - this file owns only things that call plt.savefig()/plt.show().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_comparison(metrics_df, output_dir):
    """Create bar chart comparing key metrics."""

    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Multi-Metric Comparison Across Recommender Systems", fontsize=16, y=1.02)

    metrics_to_plot = [
        ("NDCGK_10", "NDCG@10", "Higher is better"),
        ("RecallK_10", "Recall@10", "Higher is better"),
        ("ItemGiniK_10", "Gini Index@10", "Lower is better"),
        ("NDCGK_20", "NDCG@20", "Higher is better"),
        ("RecallK_20", "Recall@20", "Higher is better"),
        ("ItemGiniK_20", "Gini Index@20", "Lower is better"),
    ]

    for idx, (metric, title, direction) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        if metric in metrics_df.columns:
            data = metrics_df[["algorithm", metric]].copy()
            data = data.sort_values(metric, ascending=(direction == "Lower is better"))

            bars = ax.bar(range(len(data)), data[metric])

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data["algorithm"], rotation=45, ha="right")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n({direction})")
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_exposure_tier_composition(all_recommendations: dict, output_dir: str | Path) -> Path:
    """Bar chart of each algorithm's recommended-item profile broken down by
    exposure tier (recommended to exactly 1 user, 2-9, 10-99, 100+) - the
    figure behind the Discussion paragraph on what Coverage@10's gain is
    actually made of (see the paper: a wider profile is mostly additional
    items receiving repeated, non-trivial exposure, not just singletons).

    Args:
        all_recommendations: {algorithm_name: recos_df} with an "item_id"
            column - e.g. from `src.utils.scores2recommendations`.
        output_dir: Directory to save the figure into.

    Returns:
        Path to the saved PNG.
    """
    output_dir = Path(output_dir)
    tier_edges = [(1, 1, "1"), (2, 9, "2–9"), (10, 99, "10–99"), (100, np.inf, "100+")]
    tier_labels = [label for _, _, label in tier_edges]

    algorithms = list(all_recommendations.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(algorithms)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(tier_labels))
    width = 0.8 / len(algorithms)

    for i, (algo_name, color) in enumerate(zip(algorithms, colors)):
        item_counts = all_recommendations[algo_name]["item_id"].value_counts().to_numpy()
        n_items = len(item_counts)
        shares = [np.mean((item_counts >= lo) & (item_counts <= hi)) * 100 for lo, hi, _ in tier_edges]

        offset = (i - (len(algorithms) - 1) / 2) * width
        bars = ax.bar(x + offset, shares, width, label=f"{algo_name} (n={n_items} items)", color=color)
        for bar, val in zip(bars, shares):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\nrecs/item" for label in tier_labels])
    ax.set_ylabel("Share of that algorithm's recommended items (%)")
    ax.set_title("What a wider item profile is made of")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filename = output_dir / "exposure_tier_composition.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename
