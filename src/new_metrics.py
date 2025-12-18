import numpy as np
from scipy.sparse import csr_matrix

from recpack.metrics.base import ListwiseMetricK


def gini_index(x: np.ndarray) -> float:
    """
    Compute the Gini index of a non-negative vector.
    """
    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)

    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

class ItemGiniK(ListwiseMetricK):
    """
    Computes the Gini index of the item distribution in Top-K recommendations,
    exactly like calculate_item_gini, but usable in the pipeline.
    """

    def __init__(self, K: int):
        super().__init__(K)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        """
        Compute the global Gini index over items in Top-K recommendations.
        """
        # y_pred_top_K is a sparse matrix: rows = users, cols = items
        _, item_indices = y_pred_top_K.nonzero()  # indices of recommended items

        if len(item_indices) == 0:
            gini = 1.0
        else:
            # Count how many times each item was recommended
            item_counts = np.bincount(item_indices)
            gini = gini_index(item_counts[item_counts > 0])

        # RecPack expects per-user scores; replicate global value
        self.scores_ = csr_matrix(np.full((y_pred_top_K.shape[0], 1), gini))


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
        self.scores_ = csr_matrix(
            np.full((y_pred_top_K.shape[0], 1), mean_pop)
        )


# class PublisherGiniK(ListwiseMetricK):
#     """
#     Computes the Gini index of the publisher distribution
#     in Top-K recommendations.

#     This metric is:
#     - 0 when all publishers are recommended equally often
#     - maximized when all recommendations come from one publisher

#     Lower values indicate fairer publisher exposure.

#     Requires a mapping from item indices to publisher indices.

#     :param K: Size of the recommendation list consisting of the Top-K item predictions.
#     :type K: int
#     """

#     def __init__(self, K: int):
#         super().__init__(K)
#         self.item_to_publisher = None
#         self.n_publishers = None

#     def fit(self, item_to_publisher: np.ndarray) -> None:
#         """
#         Fit an item → publisher mapping.

#         :param item_to_publisher: Array mapping item indices to publisher indices.
#                                   Length must equal number of items.
#         :type item_to_publisher: np.ndarray
#         """
#         self.item_to_publisher = np.asarray(item_to_publisher)
#         self.n_publishers = int(self.item_to_publisher.max()) + 1

#     def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
#         """
#         Compute the publisher Gini index over Top-K recommendations.
#         """

#         if self.item_to_publisher is None:
#             raise RuntimeError("PublisherGiniK must be fitted before calling calculate().")

#         # Count item recommendations
#         item_counts = np.asarray(y_pred_top_K.sum(axis=0)).ravel()

#         if item_counts.sum() == 0:
#             gini = 1.0
#         else:
#             # Aggregate item counts to publisher counts
#             publisher_counts = np.zeros(self.n_publishers, dtype=np.float64)
#             for item_idx, count in enumerate(item_counts):
#                 if count > 0:
#                     publisher_idx = self.item_to_publisher[item_idx]
#                     publisher_counts[publisher_idx] += count

#             gini = gini_index(publisher_counts[publisher_counts > 0])

#         # Replicate global score per user (RecPack convention)
#         self.scores_ = csr_matrix(
#             np.full((y_pred_top_K.shape[0], 1), gini)
#         )