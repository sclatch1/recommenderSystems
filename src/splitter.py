import logging
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from recpack.matrix import InteractionMatrix
from recpack.scenarios.splitters import Splitter

logger = logging.getLogger(__name__)


class PPACEqualExposureSplitter(Splitter):
    """Two-stage split reverse-engineered from the PPAC paper's actual
    released code/data (github.com/Stevenn9981/PPAC, dataset/ml-1M/*).

    Stage 1 - natural per-user holdout: each user's interactions are
    shuffled and a fraction is held out to test/val, the rest stays in
    train (the paper uses a fixed count of 30 per user; here it's a
    fraction, since this generalizes across datasets of different
    density). This is the *only* stage that determines `train`.

    Stage 2 - equal-exposure resample of the held-out pools: within the
    test pool and val pool independently, an item is only kept if the
    pool has at least a quota's worth of interactions for it, and is
    then downsampled to exactly that many. Items with too few pool
    interactions, and the excess above quota for items with more, are
    dropped from that eval set - they are not returned to train, mirroring
    balance_ratings.test being an independent resample of ratings.test
    rather than a different train/test boundary.

    :param test_frac: Fraction of each user's interactions held out to
        the test pool (stage 1). Defaults to 0.1.
    :type test_frac: float, optional
    :param validation_frac: Fraction of each user's interactions held out
        to the validation pool (stage 1). Defaults to 0.1.
    :type validation_frac: float, optional
    :param exposure_frac: Fraction of the total user base an item's raw
        popularity must reach to be eligible for eval (stage 2). Defaults
        to 0.1242 (matches the paper's exposure_per_item=75 on ml-1M).
    :type exposure_frac: float, optional
    :param seed: Seed for random generator for reproducibility.
    :type seed: int, optional
    :param resample_validation: If True (default), the validation pool also
        goes through stage 2's equal-exposure resample, same as test - this
        preserves the original behaviour and is what the paper's own
        protocol does. If False, validation stays as stage 1's natural
        per-user holdout (broad, full-catalog), while test is still
        equal-exposure resampled. Use False when validation drives
        early-stopping/hyperparameter selection rather than final reporting:
        on a long-tailed catalog (e.g. Steam), the equal-exposure resample
        can shrink validation to a handful of the most popular items (on
        Steam, ~1% of the catalog), which then rewards whichever checkpoint
        is *most* popularity-biased - typically the least-trained one -
        since an undertrained MF model's default behaviour is to rank
        globally-popular items highly for everyone. That's fine for a
        deliberately bias-free *test* metric, but it means early stopping
        picks the most popularity-biased checkpoint, not the best one.
    :type resample_validation: bool, optional
    :param resample_test: If True (default), the test pool goes through
        stage 2's equal-exposure resample, giving the paper's debiased
        evaluation protocol. If False, test stays as stage 1's natural
        per-user holdout instead - i.e. a standard weak-generalization
        split, which preserves the dataset's natural popularity skew in the
        test set. Stage 1 (and therefore the training set) is identical
        either way; this only changes how the test set is built from the
        held-out pool. Use False to compare against the natural-split
        protocol directly, with everything else (train set, seed, negative
        sampling) held fixed.
    :type resample_test: bool, optional
    """

    def __init__(
        self,
        test_frac: float = 0.1,
        validation_frac: float = 0.1,
        exposure_frac: float = 0.1242,
        seed: int = None,
        resample_validation: bool = True,
        resample_test: bool = True,
    ):
        super().__init__()
        self.test_frac = test_frac
        self.validation_frac = validation_frac
        self.train_frac = 1 - test_frac - validation_frac

        if self.train_frac <= 0:
            raise ValueError(f"test_frac ({test_frac}) + validation_frac ({validation_frac}) must be less than 1.0")

        self.exposure_frac = exposure_frac
        self.resample_validation = resample_validation
        self.resample_test = resample_test

        if seed is None:
            seed = np.random.get_state()[1][0]
        self.seed = seed

    def _resample_equal_exposure(self, indices, items, quota, rng):
        """Keep only items with >= quota interactions in this pool,
        downsampled to exactly quota each."""
        item_to_pool = defaultdict(list)
        for idx in indices:
            item_to_pool[items[idx]].append(idx)

        kept = []
        for pool in item_to_pool.values():
            if len(pool) < quota:
                continue
            pool = list(pool)
            rng.shuffle(pool)
            kept.extend(pool[:quota])
        return kept

    def split(self, data: InteractionMatrix) -> tuple[InteractionMatrix, InteractionMatrix, InteractionMatrix]:
        """Splits data via the paper's two-stage protocol: a natural
        per-user holdout, then an equal-exposure resample of the held-out
        pools.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 3-tuple containing (train, validation, test) matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix, InteractionMatrix]
        """
        rng = np.random.default_rng(self.seed)

        sp_mat = data.values
        users, items = sp_mat.nonzero()

        # Stage 1: natural per-user holdout.
        user_to_indices = defaultdict(list)
        for idx, u in enumerate(users):
            user_to_indices[u].append(idx)

        test_pool, val_pool, train_indices = [], [], []
        for idxs in user_to_indices.values():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n = len(idxs)
            n_test = int(np.floor(n * self.test_frac))
            n_val = int(np.floor(n * self.validation_frac))

            test_pool.extend(idxs[:n_test])
            val_pool.extend(idxs[n_test : n_test + n_val])
            train_indices.extend(idxs[n_test + n_val :])

        # Stage 2: equal-exposure resample of each held-out pool. The quota
        # is derived from exposure_frac and this dataset's actual user
        # count, so it represents the same relative popularity bar
        # regardless of dataset size (see class docstring). Items below
        # quota, and the excess above quota, are dropped here - not added
        # back to train.
        n_users = sp_mat.shape[0]
        test_quota = max(1, round(self.exposure_frac * n_users * self.test_frac))
        val_quota = max(1, round(self.exposure_frac * n_users * self.validation_frac))

        test_indices = (
            self._resample_equal_exposure(test_pool, items, test_quota, rng) if self.resample_test else test_pool
        )
        val_indices = (
            self._resample_equal_exposure(val_pool, items, val_quota, rng) if self.resample_validation else val_pool
        )

        def _build_csr(indices):
            if len(indices) == 0:
                return sp.csr_matrix(sp_mat.shape)
            rows = users[indices]
            cols = items[indices]
            return sp.csr_matrix((np.ones(len(indices)), (rows, cols)), shape=sp_mat.shape)

        train_matrix = InteractionMatrix.from_csr_matrix(_build_csr(train_indices))
        val_matrix = InteractionMatrix.from_csr_matrix(_build_csr(val_indices))
        test_matrix = InteractionMatrix.from_csr_matrix(_build_csr(test_indices))

        val_items = np.unique(val_matrix.values.nonzero()[1])
        test_items = np.unique(test_matrix.values.nonzero()[1])

        val_quota_desc = f"val_quota={val_quota}" if self.resample_validation else "val_quota=n/a (natural holdout)"
        test_quota_desc = f"test_quota={test_quota}" if self.resample_test else "test_quota=n/a (natural holdout)"
        logger.info(
            f"{self.identifier} - exposure_frac={self.exposure_frac} on {n_users} users -> "
            f"{val_quota_desc}, {test_quota_desc}. "
            f"Train: {train_matrix.values.nnz} interactions, "
            f"Val: {val_matrix.values.nnz} interactions ({len(val_items)} items), "
            f"Test: {test_matrix.values.nnz} interactions ({len(test_items)} items)"
        )

        return train_matrix, val_matrix, test_matrix
