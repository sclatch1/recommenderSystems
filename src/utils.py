from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from recpack.util import get_top_K_ranks

from src.metrics import calculate_item_gini, calculate_mean_popularity


def log_interaction_stats(name: str, X) -> None:
    """
    Print interaction-matrix statistics - shape, density, and per-user/
    per-item interaction counts. Call once before a split (on the full
    matrix) and once per output (train/val/test) after, to see exactly
    what the splitter did: e.g. a matrix's shape doesn't shrink after a
    split (row/col dimensions are preserved), only which rows/cols have
    any nonzero entries - so "N with >=1 interaction" is the number that
    actually changes.

    :param name: Label for this matrix in the printed output.
    :param X: An InteractionMatrix, or anything exposing a `.values`
        scipy sparse matrix of the same shape (falls back to treating X
        itself as sparse if there's no `.values`).
    """
    mat = X.values.tocsr() if hasattr(X, "values") else X.tocsr()
    n_users, n_items = mat.shape
    n_interactions = mat.nnz
    density = n_interactions / (n_users * n_items) * 100 if n_users and n_items else 0.0

    per_user = np.asarray(mat.sum(axis=1)).ravel()
    per_item = np.asarray(mat.sum(axis=0)).ravel()
    per_user = per_user[per_user > 0]
    per_item = per_item[per_item > 0]

    print(f"\n--- {name} ---")
    print(f"users:        {n_users} ({len(per_user)} with >=1 interaction)")
    print(f"items:        {n_items} ({len(per_item)} with >=1 interaction)")
    print(f"interactions: {n_interactions}")
    print(f"density:      {density:.4f}%")
    if len(per_user):
        print(
            f"interactions/user  min={per_user.min():.0f} median={np.median(per_user):.0f} "
            f"mean={per_user.mean():.1f} max={per_user.max():.0f}"
        )
    if len(per_item):
        print(
            f"interactions/item  min={per_item.min():.0f} median={np.median(per_item):.0f} "
            f"mean={per_item.mean():.1f} max={per_item.max():.0f}"
        )


def aggregate_metrics_over_seeds(
    df_metrics_per_seed: pd.DataFrame, algorithm_col: str = "algorithm", group_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Aggregate per-seed metrics into mean +/- std per algorithm (or per
    algorithm/split, etc., if `group_cols` is given).

    research-plan.md's repeatability protocol calls for running experiments
    over multiple random seeds and reporting mean and standard deviation
    before treating any comparison as final - this computes that aggregate
    from the per-seed results (e.g. `src.pipeline.run_pipeline` with several
    `--seeds`).

    :param df_metrics_per_seed: One row per (algorithm, seed) - e.g. the
        concatenation of `pipe.get_metrics(short=True).reset_index(names="algorithm")`
        across several seeded pipeline runs, each tagged with its own "seed"
        column.
    :param algorithm_col: Column identifying each row's algorithm. Ignored if
        `group_cols` is given.
    :param group_cols: Columns to group by before aggregating, e.g.
        `["algorithm", "split"]` when results span more than one evaluation
        protocol and seeds must not be averaged across protocols. Defaults
        to `[algorithm_col]`.
    :return: One row per group, with a `<metric>_mean`/`<metric>_std`
        column pair for every numeric metric column, plus `n_seeds`. `seed`
        itself isn't a metric, so it's excluded from the aggregated columns.
        std is NaN where n_seeds == 1 (pandas' default, ddof=1).
    """
    if group_cols is None:
        group_cols = [algorithm_col]

    metric_cols = df_metrics_per_seed.select_dtypes(include="number").columns.drop("seed", errors="ignore")

    grouped = df_metrics_per_seed.groupby(group_cols)[metric_cols]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = grouped.size()

    return agg.reset_index()


def add_pct_change_vs_baseline(
    df_metrics: pd.DataFrame, baseline: str, algorithm_col: str = "algorithm"
) -> pd.DataFrame:
    """
    Add a "<metric>_pct_change" column per numeric metric column, computed
    as (value - baseline_value) / baseline_value * 100 relative to the row
    where algorithm_col == baseline.

    This is a raw percent change, not adjusted for metric direction - a
    negative pct_change on a lower-is-better metric (Gini, PRU, PPRU,
    MeanPopK) means an improvement, same as a plain reading of "X% lower"
    would. The baseline's own row is 0% on every metric by construction.

    :param df_metrics: Metrics table with one row per algorithm and one
        column per metric - e.g. pipe.get_metrics(short=True)
        .reset_index(names="algorithm").
    :param baseline: Value in `algorithm_col` to treat as the 0% reference
        (must be present exactly once).
    :param algorithm_col: Column identifying each row's algorithm.
    :raises ValueError: if `baseline` isn't found in df_metrics[algorithm_col].
    """
    if baseline not in df_metrics[algorithm_col].values:
        raise ValueError(f"Baseline algorithm {baseline!r} not found among {df_metrics[algorithm_col].tolist()}")

    baseline_row = df_metrics.loc[df_metrics[algorithm_col] == baseline].iloc[0]
    metric_cols = df_metrics.select_dtypes(include="number").columns

    out = df_metrics.copy()
    for col in metric_cols:
        baseline_val = baseline_row[col]
        out[f"{col}_pct_change"] = (df_metrics[col] - baseline_val) / baseline_val * 100

    return out


# Helper: convert sparse top-K rank matrix to dataframe (user,item,rank)
def matrix2df(X) -> pd.DataFrame:
    coo = sp.sparse.coo_array(X)
    return pd.DataFrame({"user_id": coo.row, "item_id": coo.col, "value": coo.data})


def save_metrics_incremental(dir_path, df_metrics, df_optimization=None, prefix="test", suffix=".json"):
    """
    Save df_metrics and df_optimization to the next available file names like:
    df_metrics: test_0.json, test_1.json, ...
    df_optimization: test_optimize_0.json, test_optimize_1.json, ...

    Args:
        dir_path (str or Path): Directory where the files should be saved.
        df_metrics (pd.DataFrame): Metrics DataFrame to save.
        df_optimization (pd.DataFrame, optional): Optimization DataFrame to save.
        prefix (str): File prefix (default: "test")
        suffix (str): File suffix (default: ".json")
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # --- For df_metrics ---
    existing_metrics_indices = []
    for path in dir_path.glob(f"{prefix}_*{suffix}"):
        if "optimize" in path.stem:
            continue
        try:
            idx = int(path.stem.split("_")[-1])
            existing_metrics_indices.append(idx)
        except ValueError:
            continue

    next_metrics_idx = max(existing_metrics_indices) + 1 if existing_metrics_indices else 0
    output_path = dir_path / f"{prefix}_{next_metrics_idx}{suffix}"
    df_metrics.to_json(output_path, indent=4)

    # --- For df_optimization ---
    if df_optimization is not None:
        existing_opt_indices = []
        for path in dir_path.glob(f"{prefix}_optimize_*{suffix}"):
            try:
                idx = int(path.stem.split("_")[-1])
                existing_opt_indices.append(idx)
            except ValueError:
                continue

        next_opt_idx = max(existing_opt_indices) + 1 if existing_opt_indices else 0
        output_path_o = dir_path / f"{prefix}_optimize_{next_opt_idx}{suffix}"
        df_optimization.to_json(output_path_o, indent=4)

    return output_path


def save_recommendations_incremental(dir_path, df_recos, algorithm="algo", suffix=".csv"):
    """
    Save df_recos to the next available CSV file like:
    algo_0.csv, algo_1.csv, ...

    Args:
        dir_path (str or Path): Directory to save the files.
        df_recos (pd.DataFrame): Recommendations dataframe.
        algorithm (str): Algorithm name or prefix for the file.
        suffix (str): File suffix (default: ".csv")
    Returns:
        Path: Path to the saved file.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Find existing files for this algorithm
    existing_indices = []
    for path in dir_path.glob(f"{algorithm}_*{suffix}"):
        try:
            idx = int(path.stem.split("_")[-1])
            existing_indices.append(idx)
        except ValueError:
            continue

    next_idx = max(existing_indices) + 1 if existing_indices else 0
    output_path = dir_path / f"{algorithm}_{next_idx}{suffix}"

    lowercase_output_path = Path(str(output_path).lower())

    df_recos.to_csv(lowercase_output_path, index=False)
    return output_path


def scores2recommendations(
    scores: sp.sparse.csr_matrix,
    X_test_in: sp.sparse.csr_matrix,
    recommendation_count: int,
    user_id_mapping: dict,
    item_id_mapping: dict,
    prevent_history_recos=True,
) -> pd.DataFrame:
    # ensure you don't recommend fold-in items
    if prevent_history_recos:
        scores[(X_test_in > 0)] = 0

    # rank items
    ranks = get_top_K_ranks(scores, recommendation_count)

    # convert to a dataframe with re-indexed IDs
    df_recos = matrix2df(ranks).rename(columns={"value": "rank"})

    # Create reverse mapping: {reindexed_id: original_id}
    reverse_user_mapping = {v: k for k, v in user_id_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_id_mapping.items()}

    # Apply the reverse mapping
    df_recos["user_id"] = df_recos["user_id"].map(reverse_user_mapping)
    df_recos["item_id"] = df_recos["item_id"].map(reverse_item_mapping)

    df_recos = df_recos.sort_values(["user_id", "rank"])

    return df_recos


# ============================================================================
# Comparative analysis & reporting - operate on {algorithm_name: recos_df}
# dicts and a metrics DataFrame produced by a PipelineBuilder run, not on
# any single algorithm/dataset. The plotting counterpart (plot_metrics_
# comparison) lives in src/plotting.py.
# ============================================================================


def calculate_coverage(recommendations_df, total_items):
    """Calculate catalog coverage."""
    unique_items = recommendations_df["item_id"].nunique()
    return unique_items / total_items * 100


def calculate_popularity_novelty(recommendations_df, train_df):
    """
    Mean self-information novelty (-log2 popularity) of recommended items,
    from training-set popularity. Distinct from metrics.calculate_novelty,
    which measures distance-from-history rather than rarity.
    """
    item_popularity = train_df.groupby("item_id").size()
    total_interactions = len(train_df)

    novelties = []
    for item_id in recommendations_df["item_id"]:
        if item_id in item_popularity.index:
            pop = item_popularity[item_id] / total_interactions
            novelty = -np.log2(pop)
            novelties.append(novelty)

    return np.mean(novelties) if novelties else 0


def get_popularity_bins(train_df, n_bins=3):
    """Divide items into popularity bins (head, torso, tail)."""
    item_popularity = train_df.groupby("item_id").size().sort_values(ascending=False)
    n_items = len(item_popularity)

    bins = {
        "head": set(item_popularity.index[: int(n_items * 0.2)]),
        "torso": set(item_popularity.index[int(n_items * 0.2) : int(n_items * 0.5)]),
        "tail": set(item_popularity.index[int(n_items * 0.5) :]),
    }
    return bins, item_popularity


def analyze_popularity_distribution(recommendations_df, popularity_bins):
    """Analyze distribution of recommendations across popularity bins."""
    recommended_items = set(recommendations_df["item_id"].unique())

    results = {
        "head": len(recommended_items & popularity_bins["head"]),
        "torso": len(recommended_items & popularity_bins["torso"]),
        "tail": len(recommended_items & popularity_bins["tail"]),
    }

    total = sum(results.values())
    if total > 0:
        results = {k: v / total * 100 for k, v in results.items()}

    return results


def compute_long_tail_coverage(all_recommendations, train_df):
    """
    For each algorithm, what share of its *distinct* recommended items
    (not weighted by how often each was recommended) falls into the head
    (top 20% most popular items in train_df), torso (next 30%), or tail
    (remaining 50%) - see get_popularity_bins/analyze_popularity_distribution.

    :param all_recommendations: {algorithm_name: recommendations_df} - e.g.
        the dict built in src/pipeline.py's `_run_analysis`.
    :param train_df: Training interactions, used to rank items by popularity.
    :return: One row per algorithm, with "Algorithm", "Head (%)",
        "Torso (%)", "Tail (%)" columns.
    """
    popularity_bins, _ = get_popularity_bins(train_df)

    results = []
    for algo_name, recs_df in all_recommendations.items():
        distribution = analyze_popularity_distribution(recs_df, popularity_bins)
        results.append(
            {
                "Algorithm": algo_name,
                "Head (%)": distribution["head"],
                "Torso (%)": distribution["torso"],
                "Tail (%)": distribution["tail"],
            }
        )

    return pd.DataFrame(results)


def create_comprehensive_metrics_table(all_recommendations, train_df, test_df, k=10, metrics_df=None):
    """Create comprehensive metrics comparison table.

    metrics_df, if given, is the algorithm-level metrics table from
    `pipe.get_metrics(short=True)` (see src/pipeline.py's `_run_analysis`) -
    used to pull in PRU@k/PPRU@k. Unlike Gini/Coverage/Novelty/Mean
    Popularity, those can't be recomputed from `recs_df` here: they need the
    recpack-internal (not re-mapped-to-original-id) score matrix plus the
    item/personal popularity arrays, none of which this DataFrame-based
    table has access to - they're pulled from the pipeline's own already-
    computed values instead.
    """
    results = []
    total_items = train_df["item_id"].nunique()
    metrics_by_algo = metrics_df.set_index("algorithm") if metrics_df is not None else None

    for algo_name, recs_df in all_recommendations.items():
        gini = calculate_item_gini(recs_df, k=k, total_items=total_items)
        coverage = calculate_coverage(recs_df, total_items)
        novelty = calculate_popularity_novelty(recs_df, train_df)
        mean_popularity = calculate_mean_popularity(recs_df, train_df, k)

        item_counts = recs_df["item_id"].value_counts()
        avg_recs_per_item = item_counts.mean()
        std_recs_per_item = item_counts.std()

        row = {
            "Algorithm": algo_name,
            "Gini Index": gini,
            "Coverage (%)": coverage,
            "Novelty": novelty,
            "Mean Popularity": mean_popularity,
            "Avg Recs/Item": avg_recs_per_item,
            "Std Recs/Item": std_recs_per_item,
            "Unique Items": len(item_counts),
        }

        if metrics_by_algo is not None:
            row["PRU"] = metrics_by_algo.loc[algo_name, f"PRUK_{k}"]
            row["PPRU"] = metrics_by_algo.loc[algo_name, f"PPRUK_{k}"]

        results.append(row)

    return pd.DataFrame(results)


def generate_analysis_report(metrics_df, comprehensive_metrics, popularity_distribution, output_dir: str):
    """Generate a text report summarizing findings."""
    output_dir = Path(output_dir)
    report_path = output_dir / "analysis_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATION SYSTEM ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ACCURACY METRICS (NDCG@10)\n")
        f.write("-" * 40 + "\n")
        f.writelines(f"{row['algorithm']:20s}: {row['NDCGK_10']:.4f}\n" for _, row in metrics_df.iterrows())
        f.write("\n")

        f.write("2. FAIRNESS METRICS (Gini Index@10 - Lower is Better)\n")
        f.write("-" * 40 + "\n")
        f.writelines(f"{row['algorithm']:20s}: {row['ItemGiniK_10']:.4f}\n" for _, row in metrics_df.iterrows())
        f.write("\n")

        f.write("3. DIVERSITY METRICS\n")
        f.write("-" * 40 + "\n")
        for _, row in comprehensive_metrics.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  Coverage: {row['Coverage (%)']:.2f}%\n")
            f.write(f"  Novelty:  {row['Novelty']:.4f}\n")
            f.write(f"  Unique Items: {row['Unique Items']}\n")
        f.write("\n")

        f.write("4. POPULARITY BIAS METRICS (Lower is Better)\n")
        f.write("-" * 40 + "\n")
        for _, row in comprehensive_metrics.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  Mean Popularity: {row['Mean Popularity']:.4f}\n")
            if "PRU" in row:
                f.write(f"  PRU:             {row['PRU']:.4f}\n")
            if "PPRU" in row:
                f.write(f"  PPRU:            {row['PPRU']:.4f}\n")
        f.write("\n")

        f.write("5. LONG-TAIL COVERAGE\n")
        f.write("-" * 40 + "\n")
        for _, row in popularity_distribution.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  Head:  {row['Head (%)']:.2f}%\n")
            f.write(f"  Torso: {row['Torso (%)']:.2f}%\n")
            f.write(f"  Tail:  {row['Tail (%)']:.2f}%\n")
        f.write("\n")

        f.write("6. KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")

        best_acc = metrics_df.loc[metrics_df["NDCGK_10"].idxmax()]
        f.write(f"- Best Accuracy: {best_acc['algorithm']} (NDCG@10: {best_acc['NDCGK_10']:.4f})\n")

        best_fair = metrics_df.loc[metrics_df["ItemGiniK_10"].idxmin()]
        f.write(f"- Best Fairness: {best_fair['algorithm']} (Gini: {best_fair['ItemGiniK_10']:.4f})\n")

        best_cov = comprehensive_metrics.loc[comprehensive_metrics["Coverage (%)"].idxmax()]
        f.write(f"- Best Coverage: {best_cov['Algorithm']} ({best_cov['Coverage (%)']:.2f}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"\nAnalysis report saved to: {report_path}")
