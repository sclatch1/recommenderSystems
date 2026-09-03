import numpy as np
import pandas as pd
import torch
from recpack.metrics.base import ListwiseMetricK
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr


def gini_index(x: np.array):
    """
    src: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797
    :param x: a numpy array of the values
    :return: gini index of array of values
    """
    n = len(x)
    if n <= 1:
        return 0.0
    sorted_x = np.sort(x)
    x_cum = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(x_cum) / x_cum[-1]) / n


def get_top_k(predictions: pd.DataFrame, k: int):
    """
    Processes recommendations so only the first k unique recommendation per user are retained.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    return predictions.drop_duplicates(["user_id", "item_id"]).groupby("user_id").head(k).reset_index(drop=True)


def calculate_user_coverage(predictions: pd.DataFrame, k: int, n_users: int) -> float:
    """
    Calculates the fraction of users who received at least k unique recommendations.

    This metric is maximally 1 when at least k items are recommended for each user.
    This metric is minimally 0 when no recommendations are given.
    For fair recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    n_users: int
        Target number of users.
        Must be at least 1.
    """

    if n_users < 1:
        raise ValueError("n_users mut be at least 1.")

    recommendations = get_top_k(predictions, k)
    recommendations_per_user = recommendations["user_id"].value_counts()
    return (recommendations_per_user >= k).sum() / n_users


def calculate_item_coverage(predictions: pd.DataFrame, k: int, n_items: int) -> float:
    """
    Calculates the fraction of items that received at least one recommendation.

    This metric is maximally 1 when each item is recommended at least once.
    This metric is minimally 1/n_items when all recommendations are the same item.
    For diverse and fair recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    n_items: int
        Number of unique items that are recommendable.
        Must be at least 1.
    """

    if n_items < 1:
        raise ValueError("n_items must be at least 1")

    recommendations = get_top_k(predictions, k)
    return recommendations["item_id"].nunique() / n_items


def calculate_item_gini(predictions: pd.DataFrame, k: int, total_items: int) -> float:
    """
    Calculates the gini index of the item distribution over the whole catalog.

    Items that are never recommended count as zero exposure - they are NOT
    excluded from the calculation. Excluding them would score "same item to
    every user" (the worst possible concentration) as perfectly equal
    (gini=0), since with only one item ever appearing, there'd be nothing to
    be unequal relative to. gini_index only depends on the multiset of
    counts, not which item each one belongs to, so zero-padding by count
    (`total_items - len(item_counts)` zeros) is equivalent to reindexing
    against the real catalog id set, and simpler.

    This metric is maximally 1 - 1/total_items when all recommendations are the same item.
    This metric is minimally 0 when every item is recommended in equal amounts.
    For fair recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    total_items: int
        Total number of items in the catalog (e.g. `train_df["item_id"].nunique()`),
        not just the items that appear in `predictions` - items with zero
        recommendations must still count toward the distribution.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    item_counts = recommendations["item_id"].value_counts().to_numpy()
    num_unrecommended = max(total_items - len(item_counts), 0)
    counts = np.concatenate([item_counts, np.zeros(num_unrecommended, dtype=item_counts.dtype)])
    return gini_index(counts)


def calculate_mean_popularity(
    predictions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    k: int,
) -> float:
    """
    Mean popularity of recommended items.

    Lower is better (less popularity bias).
    """
    # Count item popularity from training data
    item_popularity = train_interactions["item_id"].value_counts()

    # Keep top-k per user
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 0.0

    # Map popularity (unseen items → 0)
    pop = recommendations["item_id"].map(item_popularity).fillna(0)

    # Log-transform to reduce heavy-tail effect
    return np.mean(np.log1p(pop.to_numpy()))


def calculate_publisher_gini(predictions: pd.DataFrame, k: int, item_mapper: pd.Series) -> float:
    """
    Calculates the gini index of the publisher distribution.

    This metric is maximally 1 - 1/n_publishers when all recommendations are the same publisher.
    This metric is minimally 0 when every publisher is recommended in equal amounts.
    For fair recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    item_mapper: pandas.Series
        Mapping from item identifiers to publishers.
        The index contains item identifiers, and the values are publisher identifiers (e.g., strings).
        Must have an entry for each item occurring in predictions.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    publisher_counts = item_mapper[recommendations["item_id"]].value_counts()
    return gini_index(publisher_counts.to_numpy())


def calculate_calibrated_recall(predictions: pd.DataFrame, k: int, test_out: pd.DataFrame) -> float:
    """
    Calculates the average calibrated recall@k across all test users.

    The calibrated recall for a user is the number of relevant recommendations (hits) divided by the minimum of k and the number of relevant items for that user.
    The calibrated recall differs from the standard recall because it takes into account that you cannot have more than k hits.

    This metric is maximally 1 when for each user either all recommendation are relevant or all relevant items are among the recommendations.
    This metric is minimally 0 when none of the recommendations are relevant.
    For accurate recommendations, higher is better.

    See https://recpack.froomle.ai/generated/recpack.metrics.CalibratedRecallK.html for more information.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_out: pandas.DataFrame
        A dataframe of ground truth user–item interactions.
        Must contain at least one interaction.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    Notes
    -----
    Users that are not in test_out are ignored.
    """
    if len(test_out) == 0:
        raise ValueError("test_out must contain at least one interaction.")

    recommendations = get_top_k(predictions, k)

    # count the relevant recommendation per user
    hits = pd.merge(recommendations, test_out, on=["user_id", "item_id"])
    hits_per_user = hits["user_id"].value_counts()

    # count the relevant items per user
    relevant_per_user = test_out["user_id"].value_counts()
    calibrated_relevant_per_user = np.minimum(relevant_per_user, k)

    # hits_per_user.user_id is subset of calibrated_relevant_per_user.user_id
    # fill_value=0 ensures we handle the missing users
    return hits_per_user.div(calibrated_relevant_per_user, fill_value=0).mean()


def calculate_ndcg(predictions: pd.DataFrame, k: int, test_out: pd.DataFrame) -> float:
    """
    Calculates the average NDCG@k across all test users.

    The NDCG for a user is a normalized weighted sum of relevant recommendations (hits).
    The weights are determined by the rank of the recommendation (with earlier recommendation having more weight).
    The normalization is determined so that k hits results in an NDCG of 1.

    This metric is maximally 1 when for each user either all recommendation are relevant or all relevant items are among the first recommendations.
    This metric is minimally 0 when none of the recommendations are relevant.
    For accurate recommendations, higher is better.

    See https://recpack.froomle.ai/generated/recpack.metrics.NDCGK.html for more information.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_out: pandas.DataFrane
        A dataframe of ground truth user–item interactions.
        Must contain at least one interactions
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    Notes
    -----
    Users that are not in test_out are ignored.
    """
    if len(test_out) == 0:
        raise ValueError("test_out must contain at least one interaction.")

    recommendations = get_top_k(predictions, k)

    # calculate dcg = weighted sum of hits per user
    recommendations["weight"] = 1 / np.log2(recommendations.groupby("user_id").cumcount() + 2)
    hits = pd.merge(recommendations, test_out, on=["user_id", "item_id"])
    dcg_per_user = hits.groupby("user_id")["weight"].sum()

    # calculate ideal dcg per user
    idcg_table = np.cumsum(1 / np.log2(np.arange(k) + 2))
    relevant_counts = test_out["user_id"].value_counts()
    calibrated_relevant_counts = np.minimum(relevant_counts, k)
    idcg_per_user = pd.Series(idcg_table[calibrated_relevant_counts - 1], calibrated_relevant_counts.index)

    # dcg.user_id is subset of idcg.user_id
    # fill_value=0 ensures we handle the missing users
    return dcg_per_user.div(idcg_per_user, fill_value=0).mean()


def calculate_intra_list_sim(
    predictions: pd.DataFrame,
    k: int,
    item_similarity_matrix: np.ndarray,
) -> float:
    """
    Calculates the average intra list similarity across users [1].

    The intra list similarity for a user is the average pairwise similarity of their recommendations.

    This metric is maximally 1 when for each user, all recommended items are fully similar (similarity=1).
    This metric is minimally 0 when for each user, none of the recommendations are similar (similarity=0).
    For diverse recommendations, lower is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    item_similarity_matrix: np.ndarray of shape (n_items, n_items)
        A matrix containing pairwise items similarities between 0 and 1.
        Must have entries for all items in predictions and test_in.

    Notes
    -----
    If no valid recommendations are provided, the result is 1.

    References
    ----------
    .. [1] Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen. 2005.
           Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web (WWW '05). Association for Computing Machinery, New York, NY, USA, 22–32.
           https://doi.org/10.1145/1060745.1060754
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 1

    # gather all pairwise similarities
    cross = pd.merge(recommendations, recommendations, on="user_id")
    cross = cross[cross["item_id_x"] < cross["item_id_y"]]
    cross["value"] = item_similarity_matrix[cross["item_id_x"], cross["item_id_y"]]

    # compute average pairwise similarity  per user
    ils_per_user = cross.groupby("user_id")["value"].mean()

    # average ils across users
    return ils_per_user.mean()


def calculate_novelty(
    predictions: pd.DataFrame,
    k: int,
    test_in: pd.DataFrame,
    item_distance_matrix: np.ndarray,
) -> float:
    """
    Calculates the average novelty across users [1].

    The novelty for a user is the average distance between their recommendations and their history.

    This metric is maximally 1 when for each test user, all recommended items are dissimilar to their history (distance=1).
    This metric is minimally 0 when for each test user, all recommended items are similar to their history (distance=0).
    For diverse recommendations, higher is better.

    Parameters
    ----------
    predictions: pandas.DataFrame
        A dataframe of predicted user–item interactions (recommendations).
        Ordering matters as only the first k unique predictions for each user will be retained.
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    k: int
        Number of unique recommendation to retain per user.
        Must be at least 1.

    test_in: pandas.DataFrame
        A dataframe of historical user–item interactions (fold-in).
        Must contain the following columns:

        - ``user_id`` : int
            Identifier for the user.
        - ``item_id`` : int
            Identifier for the item.

    item_distance_matrix: np.ndarray of shape (n_items, n_items)
        A matrix containing pairwise items distances between 0 and 1.
        Must have entries for all items in predictions and test_in.

    Notes
    -----
    If no valid recommendations are provided, the result is 0.

    References
    ----------
    .. [1] Hurley, N. and Zhang, M. 2011.
           Novelty and Diversity in Top-N Recommendation – Analysis and Evaluation. ACM Trans. Internet Technol. 10, 4, Article 14 (March 2011), 30 pages.
           https://doi.org/10.1145/1944339.1944341
    """
    recommendations = get_top_k(predictions, k)
    if len(recommendations) == 0:
        return 0

    # gather all pairwise similarities between predictions and history
    cross = pd.merge(recommendations, test_in, on="user_id")
    cross["value"] = item_distance_matrix[cross["item_id_x"], cross["item_id_y"]]

    # compute average similarity per user
    novelty_per_user = cross.groupby("user_id")["value"].mean()

    # average novelty across users
    return novelty_per_user.mean()


############# functions from ppac framework ########


def cal_global_nov(train_records, num_items):
    """
    Calculate global novelty scores for items

    Parameters
    ----------
    train_records : dict
        Dictionary mapping user indices to lists of item indices
    num_items : int
        Total number of items

    Returns
    -------
    torch.Tensor, torch.Tensor
        Global novelty scores and popularity counts
    """
    pop = [0] * num_items
    nov = [1] * num_items
    for user in train_records:
        for item in train_records[user]:
            pop[item] += 1
    u_num = len(train_records)
    for i in range(num_items):
        if pop[i] > 0:
            nov[i] = -(1 / np.log2(u_num)) * np.log2(pop[i] / u_num)
    return torch.tensor(nov), torch.tensor(pop)


def _jaccard_top_k_neighbors(X, X_T, user_sizes, start: int, end: int, k: int):
    """Find the k most Jaccard-similar other users for each user in
    X[start:end), as a sparse adjacency slice.

    A user's raw top-k can include "neighbors" with literally zero shared
    items whenever fewer than k other users overlap with them at all
    (common for users with very few interactions) - those are dropped, so
    the returned per-user neighbor count can be less than k.

    Returns
    -------
    scipy.sparse.csr_matrix, np.ndarray
        (n_batch_users x num_users) neighbor adjacency (1 = real neighbor,
        i.e. nonzero Jaccard overlap), and each batch user's real-neighbor
        count (<= k).
    """
    num_users = X.shape[0]
    n_batch_users = end - start

    shared_item_counts = np.asarray((X[start:end] @ X_T).todense(), dtype=np.float32)
    shared_item_counts[np.arange(n_batch_users), np.arange(start, end)] = 0  # a user is never their own neighbor

    union_sizes = user_sizes[start:end, None] + user_sizes[None, :] - shared_item_counts
    jaccard = np.divide(shared_item_counts, union_sizes, out=np.zeros_like(shared_item_counts), where=union_sizes > 0)

    top_k_idx = np.argpartition(-jaccard, k - 1, axis=1)[:, :k]  # (n_batch_users, k), unsorted
    row_idx = np.arange(n_batch_users)[:, None]
    top_k_jaccard = jaccard[row_idx, top_k_idx]
    is_real_neighbor = top_k_jaccard > 0  # drop "neighbors" with zero actual overlap

    real_neighbor_counts = is_real_neighbor.sum(axis=1)
    neighbor_adjacency = csr_matrix(
        (
            np.ones(is_real_neighbor.sum(), dtype=np.float32),
            (row_idx.repeat(k)[is_real_neighbor.ravel()], top_k_idx[is_real_neighbor]),
        ),
        shape=(n_batch_users, num_users),
    )
    return neighbor_adjacency, real_neighbor_counts


def _novelty_from_neighbor_popularity(local_pop: np.ndarray, neighbor_counts: np.ndarray) -> np.ndarray:
    """PPAC paper's local-novelty formula: -(1/log2(K)) * log2(local_pop/K),
    where K is a user's real-neighbor count - i.e. how surprising it is,
    relative to how many neighbors even *could* have interacted with an
    item, that this many of them did.

    Undefined for a (user, item) pair with no signal - the user has no real
    neighbors, or none of them interacted with the item - which both
    default to the paper's own "fully novel" value of 1.0, matching
    cal_global_nov's zero-popularity default.
    """
    safe_k = np.maximum(neighbor_counts, 2)  # avoid log2(0)/log2(1) for near-isolated users
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = local_pop / neighbor_counts[:, None]
        log_ratio = np.where(ratio > 0, np.log2(ratio, where=ratio > 0), 0.0)
        local_nov = -(1.0 / np.log2(safe_k))[:, None] * log_ratio

    has_signal = (local_pop > 0) & (neighbor_counts[:, None] > 0)
    return np.where(has_signal, local_nov, 1.0)


def cal_local_nov(X: csr_matrix, num_neighbors: int = 30, batch_size: int = 1000):
    """
    Calculate personal (local) popularity/novelty scores for user-item pairs,
    following the PPAC paper: for each user u, find the `num_neighbors` most
    Jaccard-similar users (by interaction-set overlap), and let local_pop[u, i]
    be the number of those neighbors who interacted with item i.

    Unlike a user's own interaction history, this is defined for every
    user-item pair, including items the user has never interacted with -
    which is what makes it usable as a signal for candidate recommendations.

    Computed in row-batches, fully vectorized within each batch (no
    per-user Python loop), so it scales to tens of thousands of users. Each
    batch briefly densifies its (batch_size x num_users) similarity slice;
    batch_size trades peak memory for fewer, larger vectorized ops. See
    `_jaccard_top_k_neighbors` for the per-batch neighbor selection and
    `_novelty_from_neighbor_popularity` for the final novelty formula.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Binary user-item interaction matrix, shape (num_users, num_items).
    num_neighbors : int
        Number of most similar users (by Jaccard similarity) per user.
    batch_size : int
        Number of users to process per batch.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Local novelty scores (num_users x num_items) and local popularity (num_users x num_items)
    """
    X = X.tocsr()
    num_users, num_items = X.shape
    X_T = X.T.tocsr()
    user_sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    k = min(num_neighbors, max(num_users - 1, 1))

    local_pop = np.zeros((num_users, num_items), dtype=np.float32)
    neighbor_counts = np.zeros(num_users, dtype=np.float32)

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        neighbor_adjacency, batch_neighbor_counts = _jaccard_top_k_neighbors(X, X_T, user_sizes, start, end, k)
        neighbor_counts[start:end] = batch_neighbor_counts
        local_pop[start:end] = np.asarray((neighbor_adjacency @ X).todense())

    local_nov = _novelty_from_neighbor_popularity(local_pop, neighbor_counts)

    return torch.tensor(local_nov, dtype=torch.float32), torch.tensor(local_pop, dtype=torch.float32)


def _topk_popularity_rank_scores(y_pred_top_k: csr_matrix, item_pops_for_row) -> np.ndarray:
    """
    Shared PRU/PPRU core. For each user (row) in y_pred_top_k, computes the
    negative Spearman rank correlation between that user's top-K item
    popularities (via item_pops_for_row(row, item_ids)) and the rank each
    item was given within the list.

    A user scores 0 when the correlation is undefined: fewer than 2 items
    in their top-K list, or zero popularity variance among those items.

    Parameters
    ----------
    y_pred_top_k : scipy.sparse.csr_matrix
        Sparse (n_users x n_items) matrix of 1-indexed top-K ranks, as
        produced by recpack.util.get_top_K_ranks.
    item_pops_for_row : Callable[[int, np.ndarray], np.ndarray]
        Given a row index and that row's item ids, returns their
        popularity values.
    """
    n_users = y_pred_top_k.shape[0]
    scores = np.zeros(n_users)

    for u in range(n_users):
        start, end = y_pred_top_k.indptr[u], y_pred_top_k.indptr[u + 1]
        if end - start < 2:
            continue

        item_ids = y_pred_top_k.indices[start:end]
        ranks = y_pred_top_k.data[start:end]
        pops = item_pops_for_row(u, item_ids)

        if np.all(pops == pops[0]):
            continue

        corr, _ = spearmanr(pops, ranks)
        if not np.isnan(corr):
            scores[u] = -corr

    return scores


def calculate_pru(y_pred_top_k: csr_matrix, item_popularity: np.ndarray) -> np.ndarray:
    """
    Per-user PRU@K scores, following the PPAC paper's definition:

        PRU@K = -1/|U| sum_u SRC(g_i, rank(u, i) | i in TopK(u))

    the negative Spearman rank correlation (SRC) between each recommended
    item's *global* popularity g_i and its rank within a user's top-K list,
    averaged over users. High PRU means the model is mostly just
    re-ranking by platform-wide popularity rather than personal
    preference - lower is better (less popularity-driven ranking).

    Average this function's return value over users (e.g. `.mean()`) to
    get the paper's aggregate PRU@K.

    Parameters
    ----------
    y_pred_top_k : scipy.sparse.csr_matrix
        Sparse (n_users x n_items) matrix of 1-indexed top-K ranks, as
        produced by recpack.util.get_top_K_ranks.
    item_popularity : np.ndarray
        Any monotonic proxy for g_i per item (e.g. raw interaction
        counts from training data) - Spearman correlation is invariant to
        monotonic rescaling, so this does not need to be normalized to a
        [0, 1] fraction.

    Returns
    -------
    np.ndarray
        One PRU score per user (row of y_pred_top_k), in row order.
    """
    item_popularity = np.asarray(item_popularity)
    return _topk_popularity_rank_scores(y_pred_top_k, lambda _u, item_ids: item_popularity[item_ids])


def calculate_ppru(y_pred_top_k: csr_matrix, local_popularity: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    """
    Per-user PPRU@K scores, following the PPAC paper's definition:

        PPRU@K = -1/|U| sum_u SRC(p_{u,i}, rank(u, i) | i in TopK(u))

    identical to calculate_pru, but against each user's *personal*
    popularity p_{u,i} (see cal_local_nov above) instead of platform-wide
    popularity.

    Average this function's return value over users (e.g. `.mean()`) to
    get the paper's aggregate PPRU@K.

    Parameters
    ----------
    y_pred_top_k : scipy.sparse.csr_matrix
        Sparse (n_users x n_items) matrix of 1-indexed top-K ranks.
    local_popularity : np.ndarray
        (num_users x num_items) personal popularity matrix, e.g. the
        local_pop returned by cal_local_nov - indexed by ORIGINAL user id.
    user_ids : np.ndarray
        Original user id for each row of y_pred_top_k. Needed because
        callers (e.g. recpack metrics) may drop or reindex users, so a
        row's position no longer matches local_popularity's row order.

    Returns
    -------
    np.ndarray
        One PPRU score per user (row of y_pred_top_k), in row order.
    """
    return _topk_popularity_rank_scores(y_pred_top_k, lambda u, item_ids: local_popularity[user_ids[u], item_ids])


##############################################################


class ItemGiniK(ListwiseMetricK):
    """
    Computes the Gini index of the item distribution in Top-K recommendations
    over the whole catalog, exactly like calculate_item_gini, but usable in
    the pipeline.
    """

    def __init__(self, K: int):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        """
        Compute the global Gini index over items in Top-K recommendations,
        across the whole catalog - items never recommended count as zero
        exposure (see calculate_item_gini's docstring for why excluding them
        would be wrong), using y_pred_top_K's own column count as the
        catalog size so no extra item-count parameter is needed here.
        """
        # y_pred_top_K is a sparse matrix: rows = users, cols = items
        num_items = y_pred_top_K.shape[1]
        _, item_indices = y_pred_top_K.nonzero()  # indices of recommended items

        if len(item_indices) == 0:
            gini = 1.0
        else:
            # Count how many times each item was recommended, zero-padded
            # up to the full catalog (minlength) rather than dropping
            # never-recommended items.
            item_counts = np.bincount(item_indices, minlength=num_items)
            gini = gini_index(item_counts)

        # RecPack expects per-user scores; replicate global value
        self.scores_ = csr_matrix(np.full((y_pred_top_K.shape[0], 1), gini))


class ItemCoverageK(ListwiseMetricK):
    """
    Computes the fraction of the catalog that receives at least one
    recommendation in Top-K, exactly like calculate_item_coverage but
    usable in the pipeline.

    Unlike ItemGiniK, this measures only breadth - how much
    of the catalog gets touched at all - and is insensitive to how evenly
    exposure is distributed among the items that do get recommended (e.g.
    a handful of dominant items sitting on top of a long thin tail of
    once-recommended items still counts as full coverage of that tail).
    Report alongside the Gini metrics for that reason: coverage answers
    "did debiasing widen exposure", the Gini metrics answer "is that
    exposure spread evenly" - neither answers the other's question.

    Higher is better (more of the catalog reaches at least one user).
    """

    def __init__(self, K: int):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        # y_pred_top_K is a sparse matrix: rows = users, cols = items
        num_items = y_pred_top_K.shape[1]
        _, item_indices = y_pred_top_K.nonzero()  # indices of recommended items

        coverage = len(np.unique(item_indices)) / num_items if len(item_indices) > 0 else 0.0

        # RecPack expects per-user scores; replicate global value
        self.scores_ = csr_matrix(np.full((y_pred_top_K.shape[0], 1), coverage))


class MeanItemPopularityK(ListwiseMetricK):
    """
    Mean log popularity of items in Top-K recommendations.
    Lower = less popularity bias.
    """

    def __init__(self, K: int, item_popularity: np.ndarray):
        super().__init__(K)
        self.item_popularity = item_popularity

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix):
        # indices of recommended items
        _, item_indices = y_pred_top_K.nonzero()

        if len(item_indices) == 0:
            mean_pop = 0.0
        else:
            pops = self.item_popularity[item_indices]
            mean_pop = np.mean(np.log1p(pops))

        # replicate per user (RecPack convention)
        self.scores_ = csr_matrix(np.full((y_pred_top_K.shape[0], 1), mean_pop))


class PRUK(ListwiseMetricK):
    """
    Popularity-Rank Utility (PRU@K), from the PPAC paper: negative
    Spearman rank correlation between each recommended item's global
    popularity and its rank within a user's top-K list, averaged over
    users. High PRU means the model is mostly re-ranking by platform-wide
    popularity; lower is better (less popularity-driven ranking).

    See calculate_pru above for the definition and math.
    """

    def __init__(self, K: int, item_popularity: np.ndarray):
        super().__init__(K)
        self.item_popularity = np.asarray(item_popularity)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = calculate_pru(y_pred_top_K, self.item_popularity)
        self.scores_ = csr_matrix(scores.reshape(-1, 1))


class PPRUK(ListwiseMetricK):
    """
    Personal Popularity-Rank Utility (PPRU@K), from the PPAC paper: same
    as PRUK, but against each user's *personal* popularity (their
    k-nearest similar users' interactions - see cal_local_nov above)
    instead of platform-wide popularity.

    See calculate_ppru above for the definition and math.
    """

    def __init__(self, K: int, local_popularity: np.ndarray):
        super().__init__(K)
        self.local_popularity = np.asarray(local_popularity)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = calculate_ppru(y_pred_top_K, self.local_popularity, self.user_id_map_)
        self.scores_ = csr_matrix(scores.reshape(-1, 1))
