import numpy as np
import pandas as pd
import pytest

from src.utils import aggregate_metrics_over_seeds, compute_long_tail_coverage


def test_aggregate_metrics_over_seeds_computes_mean_and_std_per_algorithm():
    """research-plan.md's repeatability protocol calls for mean +/- std across
    seeds, not just a single run's numbers - this checks the aggregation
    itself is correct, and that a non-metric column like "seed" doesn't leak
    into the aggregated metric columns."""
    df = pd.DataFrame(
        {
            "algorithm": ["BPRMF", "BPRMF", "PPAC_BPRMF", "PPAC_BPRMF"],
            "seed": [1, 2, 1, 2],
            "NDCGK_10": [0.10, 0.20, 0.30, 0.50],
        }
    )

    summary = aggregate_metrics_over_seeds(df).set_index("algorithm")

    assert summary.loc["BPRMF", "NDCGK_10_mean"] == pytest.approx(0.15)
    assert summary.loc["BPRMF", "NDCGK_10_std"] == pytest.approx(np.std([0.10, 0.20], ddof=1))
    assert summary.loc["PPAC_BPRMF", "NDCGK_10_mean"] == pytest.approx(0.40)
    assert summary.loc["BPRMF", "n_seeds"] == 2
    assert "seed_mean" not in summary.columns
    assert "seed_std" not in summary.columns


def test_aggregate_metrics_over_seeds_single_seed_gives_nan_std():
    """With only one seed there's no variance to report - std should be NaN
    (pandas' default, ddof=1) rather than silently reporting 0, which would
    misleadingly look like a real repeatability result."""
    df = pd.DataFrame({"algorithm": ["BPRMF"], "seed": [42], "NDCGK_10": [0.15]})

    summary = aggregate_metrics_over_seeds(df).set_index("algorithm")

    assert summary.loc["BPRMF", "n_seeds"] == 1
    assert np.isnan(summary.loc["BPRMF", "NDCGK_10_std"])


def test_compute_long_tail_coverage_buckets_distinct_items_by_train_popularity():
    """compute_long_tail_coverage splits a catalog's items into head (top 20%
    most popular in train_df), torso (next 30%), and tail (remaining 50%),
    then reports what % of each algorithm's *distinct* recommended items
    (not weighted by recommendation frequency) fall into each bucket.

    10 items with strictly decreasing train popularity -> head={0,1},
    torso={2,3,4}, tail={5..9} (see get_popularity_bins). "head_heavy" only
    ever recommends head items -> 100% head. "spread" recommends exactly
    one distinct item from each bucket -> an even 3-way split, regardless
    of how many hits that head item happens to have."""
    train_df = pd.DataFrame({"item_id": sum(([i] * (10 - i) for i in range(10)), [])})
    all_recommendations = {
        "head_heavy": pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1]}),
        "spread": pd.DataFrame({"user_id": [0, 1, 2], "item_id": [0, 2, 5]}),
    }

    result = compute_long_tail_coverage(all_recommendations, train_df).set_index("Algorithm")

    assert result.loc["head_heavy", ["Head (%)", "Torso (%)", "Tail (%)"]].tolist() == pytest.approx([100.0, 0.0, 0.0])
    assert result.loc["spread", ["Head (%)", "Torso (%)", "Tail (%)"]].tolist() == pytest.approx(
        [100 / 3, 100 / 3, 100 / 3]
    )
