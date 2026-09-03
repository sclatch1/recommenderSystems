import pandas as pd
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from recpack.preprocessing.preprocessors import DataFramePreprocessor


class PreprocessBlock:
    """
    Turns a raw interaction DataFrame into recpack `InteractionMatrix`
    objects: binarize/filter playtime (`to_binary`, `to_positive`), then
    restrict to active users/items and convert to matrices with consistent
    id mappings (`apply_filter`).
    """

    def __init__(self):
        self.user_id_mapping = None
        self.item_id_mapping = None
        self.new_to_old_user_id_mapping = None
        self.new_to_old_item_id_mapping = None
        self.preprocessor = None

    def to_binary(self, df: pd.DataFrame, playtime: str) -> pd.DataFrame:
        """
        Convert playtime to binary interactions (1 if playtime > 0, else 0).

        :param df: DataFrame with playtime column
        :param playtime: Column name for playtime
        :return: DataFrame with binary interactions
        """
        assert playtime in df.columns, f"{playtime} column missing in DataFrame"

        df_binary = df.copy()
        df_binary[playtime] = (df_binary[playtime] > 0).astype(int)

        assert df_binary[playtime].isin([0, 1]).all(), "Playtime conversion to binary failed"

        return df_binary

    def to_positive(self, df: pd.DataFrame, playtime: str) -> pd.DataFrame:
        """
        Filter DataFrame to only include positive interactions (playtime > 0).

        :param df: DataFrame with playtime column
        :param playtime: Column name for playtime
        :return: Filtered DataFrame with only positive interactions
        """
        assert playtime in df.columns, f"{playtime} column missing in DataFrame"

        df_positive = df[df[playtime] > 0].copy()

        assert df_positive[playtime].isin([1]).all(), "Filtering to positive interactions failed"

        return df_positive

    def apply_filter(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, user_col: str, item_col: str, is_test: bool = False
    ) -> InteractionMatrix:
        """
        Convert a pandas DataFrame to an InteractionMatrix.

        :param df: DataFrame with user, item columns
        :param user_col: Column name for user IDs
        :param item_col: Column name for item IDs
        :param is_test: If True, use the existing preprocessor fitted on training data
        :return: InteractionMatrix
        """
        self.preprocessor = DataFramePreprocessor(
            user_ix=user_col,
            item_ix=item_col,
        )

        # Restrict to active users/items (>=10 interactions), per the
        # research plan.
        min_items_per_user = MinItemsPerUser(min_items_per_user=10, item_ix=item_col, user_ix=user_col)
        min_users_per_item = MinUsersPerItem(min_users_per_item=10, item_ix=item_col, user_ix=user_col)

        df_train = min_items_per_user.apply(df_train)
        df_train = min_users_per_item.apply(df_train)

        kept_users = df_train[user_col].unique()
        kept_items = df_train[item_col].unique()
        df_test = df_test[df_test[user_col].isin(kept_users) & df_test[item_col].isin(kept_items)].copy()

        interaction_matrices = self.preprocessor.process_many(df_train, df_test)

        self.user_id_mapping = self.preprocessor._user_id_mapping
        self.item_id_mapping = self.preprocessor._item_id_mapping

        for im in interaction_matrices:
            assert isinstance(im, InteractionMatrix), "Conversion to InteractionMatrix failed"
        return interaction_matrices, self.user_id_mapping, self.item_id_mapping
