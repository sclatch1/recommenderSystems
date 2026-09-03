import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.metrics import (
    PPRUK,
    PRUK,
    ItemCoverageK,
    ItemGiniK,
    MeanItemPopularityK,
    cal_local_nov,
    calculate_item_coverage,
    calculate_item_gini,
    calculate_ppru,
    calculate_pru,
)


def test_cal_local_nov_hand_computed_example():
    """cal_local_nov previously shipped with a real bug (degenerate local
    popularity - see project memory), and had no test coverage. This locks
    in a tiny, hand-verifiable example: 3 users, 2 items, user1 shares item0
    with user0 and item1 with user2, so both of user1's neighbors are
    "real" (nonzero Jaccard overlap), while user0/user2's neighbors include
    one zero-overlap "neighbor" that gets dropped (see
    _jaccard_top_k_neighbors) - exercising both branches of that filter in
    one example.
    """
    X = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.float32,
        )
    )

    nov, pop = cal_local_nov(X, num_neighbors=2, batch_size=1000)

    # local_pop[u, i]: how many of u's real neighbors interacted with item i.
    # user0's only real neighbor is user1 (user2 has zero overlap with
    # user0, so is dropped); user1's real neighbors are user0 and user2.
    expected_pop = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    assert pop.numpy() == pytest.approx(expected_pop)

    # user0/user2 each have 1 real neighbor who has both items -> di/K=1 ->
    # novelty 0 (unsurprising - their one real neighbor already has both).
    # user1 has 2 real neighbors, only 1 of which has each item -> di/K=0.5
    # -> novelty 1 (more surprising).
    expected_nov = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    assert nov.numpy() == pytest.approx(expected_nov)


def test_cal_local_nov_defaults_to_fully_novel_with_no_signal():
    """A user with zero overlap with every other user has no real
    neighbors at all - local_pop/local_nov should fall back to the paper's
    "fully novel" default (1.0) rather than NaN/inf from dividing by zero."""
    X = csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
    )

    nov, pop = cal_local_nov(X, num_neighbors=2, batch_size=1000)

    assert pop.numpy()[0] == pytest.approx([0, 0, 0])
    assert nov.numpy()[0] == pytest.approx([1, 1, 1])
    assert np.isfinite(nov.numpy()).all()


def test_calculate_item_gini_scores_same_item_to_everyone_as_worst_case():
    """calculate_item_gini previously dropped never-recommended items from
    the distribution entirely (pandas value_counts() only lists items that
    appear at least once), so the theoretical worst case the docstring
    itself describes - every user gets the exact same single item - used to
    score 0.0 (perfectly equal), since with only one item in the frame of
    reference there was nothing to compare it against. It should now score
    near the documented maximum, 1 - 1/total_items."""
    worst_case = pd.DataFrame({"user_id": range(100), "item_id": [7] * 100})

    gini = calculate_item_gini(worst_case, k=1, total_items=1000)

    assert gini == pytest.approx(1 - 1 / 1000, abs=1e-6)


def test_calculate_item_gini_rewards_broader_catalog_coverage():
    """Concentrating all recommendations onto a small subset of a much
    larger catalog must score as MORE unequal (higher gini) than spreading
    across most of the same catalog, even if the broader spread is itself
    somewhat uneven - this is what the un-recommended-items-count-as-zero
    fix is for."""
    catalog = 1000
    narrow = pd.DataFrame({"user_id": range(500), "item_id": [1, 2, 3, 4, 5] * 100})
    broad = pd.DataFrame({"user_id": range(500), "item_id": list(range(500)) * 1})

    gini_narrow = calculate_item_gini(narrow, k=1, total_items=catalog)
    gini_broad = calculate_item_gini(broad, k=1, total_items=catalog)

    assert gini_narrow > gini_broad


def test_item_gini_k_matches_calculate_item_gini_on_full_catalog():
    """ItemGiniK (the recpack pipeline metric) reads its catalog size off
    y_pred_top_K's own column count instead of taking an extra parameter -
    check it agrees with calculate_item_gini given the same distribution."""
    num_users, num_items = 100, 1000
    item_ids = np.array([7] * num_users)
    y_pred_top_k = csr_matrix((np.ones(num_users), (np.arange(num_users), item_ids)), shape=(num_users, num_items))

    metric = ItemGiniK(K=1)
    metric._calculate(None, y_pred_top_k)
    gini_from_metric = metric.scores_.toarray()[0, 0]

    df = pd.DataFrame({"user_id": np.arange(num_users), "item_id": item_ids})
    gini_from_function = calculate_item_gini(df, k=1, total_items=num_items)

    assert gini_from_metric == pytest.approx(gini_from_function)


def test_item_coverage_k_scores_fraction_of_catalog_touched():
    """ItemCoverageK counts distinct recommended items against the full
    catalog width (y_pred_top_K's own column count), regardless of how many
    times each one was recommended. 3 users each recommended a distinct
    item out of a 10-item catalog -> 3/10, independent of repeats."""
    y_pred_top_k = csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )

    metric = ItemCoverageK(K=1)
    metric._calculate(None, y_pred_top_k)

    assert metric.scores_.toarray()[0, 0] == pytest.approx(0.3)


def test_item_coverage_k_matches_calculate_item_coverage():
    """Cross-check ItemCoverageK (recpack pipeline metric) against the
    standalone calculate_item_coverage on an equivalent distribution."""
    num_items = 10
    item_ids = np.array([0, 1, 2])
    y_pred_top_k = csr_matrix((np.ones(3), (np.arange(3), item_ids)), shape=(3, num_items))

    metric = ItemCoverageK(K=1)
    metric._calculate(None, y_pred_top_k)
    coverage_from_metric = metric.scores_.toarray()[0, 0]

    df = pd.DataFrame({"user_id": np.arange(3), "item_id": item_ids})
    coverage_from_function = calculate_item_coverage(df, k=1, n_items=num_items)

    assert coverage_from_metric == pytest.approx(coverage_from_function)


def test_mean_item_popularity_k_hand_computed():
    """MeanItemPopularityK averages log1p(popularity) over every
    recommendation slot (not unique items), so a repeated highly-popular
    item pulls the mean toward it. user0 gets item0 (popularity 0) once,
    user1 gets item2 (popularity 990) twice."""
    y_pred_top_k = csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
    )
    item_popularity = np.array([0, 10, 990])

    metric = MeanItemPopularityK(K=1, item_popularity=item_popularity)
    metric._calculate(None, y_pred_top_k)

    # mean(log1p([0, 990, 990])), item1 never recommended so excluded
    assert metric.scores_.toarray()[0, 0] == pytest.approx(4.5991430228866585)


def test_calculate_pru_is_one_when_recommendations_are_purely_popularity_ordered():
    """PRU@K negates the Spearman correlation between item popularity and
    rank, so a model that just re-ranks by descending global popularity
    (rank 1 = most popular) is perfectly anti-correlated (corr=-1) and
    scores the maximal PRU of 1 - i.e. "purely popularity-driven"."""
    y_pred_top_k = csr_matrix(np.array([[1, 2, 3]], dtype=np.float64))
    item_popularity = np.array([300, 200, 100])  # item0 most popular ... item2 least

    scores = calculate_pru(y_pred_top_k, item_popularity)

    assert scores == pytest.approx([1.0])


def test_calculate_pru_is_zero_with_no_signal():
    """A user needs >=2 recommended items with differing popularity for the
    Spearman correlation to be defined. user0 has only 1 recommended item
    (undefined correlation); user1 has 2 items but they're equally popular
    (zero variance). Both fall back to a score of 0 rather than NaN."""
    y_pred_top_k = csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [1, 2, 0],
            ],
            dtype=np.float64,
        )
    )
    item_popularity = np.array([50, 50, 999])

    scores = calculate_pru(y_pred_top_k, item_popularity)

    assert scores == pytest.approx([0.0, 0.0])


def test_pru_k_matches_calculate_pru():
    """PRUK (the recpack pipeline metric) should just delegate to
    calculate_pru."""
    y_pred_top_k = csr_matrix(np.array([[1, 2, 3]], dtype=np.float64))
    item_popularity = np.array([300, 200, 100])

    metric = PRUK(K=3, item_popularity=item_popularity)
    metric._calculate(None, y_pred_top_k)

    assert metric.scores_.toarray().ravel() == pytest.approx(calculate_pru(y_pred_top_k, item_popularity))


def test_calculate_ppru_indexes_local_popularity_by_original_user_id():
    """calculate_ppru's local_popularity is indexed by ORIGINAL user id, not
    by the row position in y_pred_top_k - recpack drops/reindexes rows (see
    Metric._eliminate_empty_users), so row 0 of y_pred_top_k might be
    original user 5. Row 0 and row 1 here map to original users 5 and 3,
    while local_popularity rows 0 and 1 (which a row-index bug would use
    instead) are deliberately flat/zero-variance - so the test only passes
    if user_ids is actually used to look up the right rows."""
    y_pred_top_k = csr_matrix(np.array([[1, 2], [2, 1]], dtype=np.float64))
    user_ids = np.array([5, 3])

    local_popularity = np.zeros((6, 2))
    local_popularity[0] = [50, 50]  # would be (mis)used by a row-index bug
    local_popularity[1] = [50, 50]  # would be (mis)used by a row-index bug
    local_popularity[5] = [300, 100]  # correct row for original user 5
    local_popularity[3] = [100, 300]  # correct row for original user 3

    scores = calculate_ppru(y_pred_top_k, local_popularity, user_ids)

    assert scores == pytest.approx([1.0, 1.0])


def test_ppru_k_matches_calculate_ppru():
    """PPRUK (the recpack pipeline metric) should just delegate to
    calculate_ppru, reading user ids off self.user_id_map_ (normally set by
    Metric._eliminate_empty_users during calculate())."""
    y_pred_top_k = csr_matrix(np.array([[1, 2], [2, 1]], dtype=np.float64))
    user_ids = np.array([5, 3])
    local_popularity = np.zeros((6, 2))
    local_popularity[5] = [300, 100]
    local_popularity[3] = [100, 300]

    metric = PPRUK(K=2, local_popularity=local_popularity)
    metric.user_id_map_ = user_ids
    metric._calculate(None, y_pred_top_k)

    assert metric.scores_.toarray().ravel() == pytest.approx(calculate_ppru(y_pred_top_k, local_popularity, user_ids))
