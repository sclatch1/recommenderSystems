import numpy as np
from typing import Tuple
from recpack.scenarios.splitters import Splitter
from recpack.matrix import InteractionMatrix
import scipy.sparse as sp

from collections import defaultdict
from tqdm import tqdm

class ItemInterventionSplitter(Splitter):
    """
    Split interactions by item such that each item contributes
    an equal number of interactions to the out split.

    This implements the "intervened test set" used in debiasing papers.
    """

    def __init__(self, out_frac=0.1, min_per_item=1, seed=None):
        """
        :param out_frac: Fraction of interactions per item to move to out
        :param min_per_item: Minimum number of interactions per item in out
        :param seed: Random seed
        """
        super().__init__()
        self.out_frac = out_frac
        self.min_per_item = min_per_item
        self.seed = seed

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        X = data.values.tocsr()

        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()

        rows_out = []
        cols_out = []

        rows_in = []
        cols_in = []

        n_users, n_items = X.shape

        for item in range(n_items):
            users = X[:, item].nonzero()[0]
            n = len(users)

            if n == 0:
                continue

            n_out = max(self.min_per_item, int(np.floor(n * self.out_frac)))
            n_out = min(n_out, n)

            rng.shuffle(users)

            users_out = users[:n_out]
            users_in = users[n_out:]

            rows_out.extend(users_out)
            cols_out.extend([item] * len(users_out))

            rows_in.extend(users_in)
            cols_in.extend([item] * len(users_in))

        data_out = InteractionMatrix(
            sp.csr_matrix(
                (np.ones(len(rows_out)), (rows_out, cols_out)),
                shape=X.shape,
            ),
            user_mapping=data.user_mapping,
            item_mapping=data.item_mapping,
        )

        data_in = InteractionMatrix(
            sp.csr_matrix(
                (np.ones(len(rows_in)), (rows_in, cols_in)),
                shape=X.shape,
            ),
            user_mapping=data.user_mapping,
            item_mapping=data.item_mapping,
        )

        return data_in, data_out




import numpy as np
from typing import Tuple
import logging
from recpack.matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class EqualExposureSplitter(Splitter):
    """Splits data into train/validation/test sets where test and validation
    sets have equal item exposure to eliminate popularity bias.
    
    This creates a counterfactual evaluation environment by ensuring all items
    receive an equal number of interactions in the test (and validation) sets,
    better reflecting true user preferences without popularity bias.
    
    :param test_frac: Fraction of interactions for test set. Defaults to 0.1.
    :type test_frac: float, optional
    :param validation_frac: Fraction of interactions for validation set. 
        Defaults to 0.1.
    :type validation_frac: float, optional
    :param seed: Seed for random generator for reproducibility.
    :type seed: int, optional
    :param min_interactions_per_item: Minimum interactions per item required
        for inclusion in test/validation. Items with fewer interactions are
        only included in training. Defaults to 2.
    :type min_interactions_per_item: int, optional
    """
    
    def __init__(
        self, 
        test_frac: float = 0.1,
        validation_frac: float = 0.1, 
        seed: int = None,
        min_interactions_per_item: int = 2
    ):
        super().__init__()
        self.test_frac = test_frac
        self.validation_frac = validation_frac
        self.train_frac = 1 - test_frac - validation_frac
        
        if self.train_frac <= 0:
            raise ValueError(
                f"test_frac ({test_frac}) + validation_frac ({validation_frac}) "
                "must be less than 1.0"
            )
        
        if seed is None:
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self.min_interactions_per_item = min_interactions_per_item
        
    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix, InteractionMatrix]:
        """Splits data ensuring equal item exposure in test and validation sets.
        
        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 3-tuple containing (train, validation, test) matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix, InteractionMatrix]
        """
        np.random.seed(self.seed)
        
        sp_mat = data.values
        users, items = sp_mat.nonzero()
        
        # Count interactions per item
        unique_items, item_counts = np.unique(items, return_counts=True)
        
        # Filter items that have enough interactions
        valid_items = unique_items[
            item_counts >= self.min_interactions_per_item
        ]
        
        if len(valid_items) == 0:
            raise ValueError(
                f"No items have at least {self.min_interactions_per_item} "
                "interactions. Cannot create equal exposure splits."
            )
        
        # Calculate target number of interactions per item for test/val
        total_interactions = sp_mat.nnz
        test_size = int(total_interactions * self.test_frac)
        val_size = int(total_interactions * self.validation_frac)
        
        # Target interactions per item (equal exposure)
        target_test_per_item = max(1, test_size // len(valid_items))
        target_val_per_item = max(1, val_size // len(valid_items))
        
        # Group interactions by item
        item_to_interactions = {}
        for idx, (u, i) in enumerate(zip(users, items)):
            if i not in item_to_interactions:
                item_to_interactions[i] = []
            item_to_interactions[i].append(idx)
        
        # Sample interactions for each split
        test_indices = []
        val_indices = []
        train_indices = []
        
        for item in valid_items:
            if item not in item_to_interactions:
                continue
                
            interactions = item_to_interactions[item]
            np.random.shuffle(interactions)
            
            # Determine how many to sample for this item
            n_test = min(target_test_per_item, len(interactions))
            n_val = min(
                target_val_per_item, 
                len(interactions) - n_test
            )
            
            # Split indices
            test_indices.extend(interactions[:n_test])
            val_indices.extend(interactions[n_test:n_test + n_val])
            train_indices.extend(interactions[n_test + n_val:])
        
        # Add items that don't have enough interactions directly to train
        for item in unique_items:
            if item not in valid_items and item in item_to_interactions:
                train_indices.extend(item_to_interactions[item])
        
        # Create masks for the splits
        test_mask = np.zeros(len(users), dtype=bool)
        val_mask = np.zeros(len(users), dtype=bool)
        train_mask = np.zeros(len(users), dtype=bool)
        
        test_mask[test_indices] = True
        val_mask[val_indices] = True
        train_mask[train_indices] = True
        
        # Create the split matrices
        test_data = data.binary_values.multiply(
            sp_mat.multiply(test_mask[np.newaxis, :].T)
        )
        val_data = data.binary_values.multiply(
            sp_mat.multiply(val_mask[np.newaxis, :].T)
        )
        train_data = data.binary_values.multiply(
            sp_mat.multiply(train_mask[np.newaxis, :].T)
        )
        
        # Convert back to InteractionMatrix
        test_matrix = InteractionMatrix(
            test_data,
            data.item_ix,
            data.user_ix,
            timestamps=None
        )
        val_matrix = InteractionMatrix(
            val_data,
            data.item_ix, 
            data.user_ix,
            timestamps=None
        )
        train_matrix = InteractionMatrix(
            train_data,
            data.item_ix,
            data.user_ix,
            timestamps=None
        )
        
        # Log statistics
        test_items = np.unique(test_matrix.values.nonzero()[1])
        val_items = np.unique(val_matrix.values.nonzero()[1])
        
        logger.info(
            f"{self.identifier} - Train: {train_matrix.values.nnz} interactions, "
            f"Val: {val_matrix.values.nnz} interactions "
            f"({len(val_items)} items), "
            f"Test: {test_matrix.values.nnz} interactions "
            f"({len(test_items)} items)"
        )
        
        return train_matrix, val_matrix, test_matrix
    



class ItemInterventionSplitter1(Splitter):
    """
    Split interactions such that each item contributes
    an equal number of interactions to the out split.
    """

    def __init__(self, out_frac=0.1, min_per_item=1, seed=None):
        super().__init__()
        self.out_frac = out_frac
        self.min_per_item = min_per_item
        self.seed = seed

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        X = data.values.tocsr()
        rng = np.random.default_rng(self.seed)

        X_out = sp.lil_matrix(X.shape, dtype=X.dtype)
        X_in = X.copy().tolil()

        n_users, n_items = X.shape

        for item in range(n_items):
            users = X[:, item].nonzero()[0]
            n = len(users)

            if n == 0:
                continue

            n_out = max(self.min_per_item, int(np.floor(n * self.out_frac)))
            n_out = min(n_out, n)

            rng.shuffle(users)
            users_out = users[:n_out]

            # move interactions to out
            for u in users_out:
                X_out[u, item] = X[u, item]
                X_in[u, item] = 0

        X_out = X_out.tocsr()
        X_in = X_in.tocsr()

        data_out = data._copy_with_new_values(X_out)
        data_in = data._copy_with_new_values(X_in)

        return data_in, data_out
    


class ItemInterventionSplitter2(Splitter):
    """
    Split interactions such that each item contributes
    an equal fraction of interactions to the out split.

    This implements the intervened (counterfactual) test set
    used for debiasing evaluation.
    """

    def __init__(self, out_frac=0.1, min_per_item=1, seed=None):
        super().__init__()
        self.out_frac = out_frac
        self.min_per_item = min_per_item
        self.seed = seed or np.random.get_state()[1][0]

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:

        # ------------------------------------------------------------------
        # 1. Map interaction_id -> item
        # ------------------------------------------------------------------
        interaction_to_item = dict(zip(
            data._df[InteractionMatrix.INTERACTION_IX],
            data._df[InteractionMatrix.ITEM_IX]
        ))

        # ------------------------------------------------------------------
        # 2. Group interactions by item
        # ------------------------------------------------------------------
        item2interactions = defaultdict(list)
        for uid, interaction_history in tqdm(data.interaction_history):
            for interaction_id in interaction_history:
                item_id = interaction_to_item[interaction_id]
                item2interactions[item_id].append(interaction_id)  # only interaction IDs

        # ------------------------------------------------------------------
        # 3. Sample per item
        # ------------------------------------------------------------------
        in_interactions = []
        out_interactions = []

        for item_id, interactions in item2interactions.items():
            n = len(interactions)
            if n == 0:
                continue

            rstate = np.random.RandomState(self.seed + item_id)
            rstate.shuffle(interactions)

            n_out = max(self.min_per_item, int(np.floor(n * self.out_frac)))
            n_out = min(n_out, n)

            out_interactions.extend(interactions[:n_out])
            in_interactions.extend(interactions[n_out:])

        # ------------------------------------------------------------------
        # 4. Build InteractionMatrix outputs (RecPack-native)
        # ------------------------------------------------------------------
        data_in = data.interactions_in(in_interactions)
        data_out = data.interactions_in(out_interactions)

        return data_in, data_out
