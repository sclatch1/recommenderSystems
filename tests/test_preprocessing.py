import pandas as pd
import pytest

from src.preprocessing import PreprocessBlock


def test_to_binary_maps_positive_playtime_to_one_and_leaves_source_untouched():
    """to_binary should collapse any positive playtime to 1 (0 stays 0),
    without mutating the caller's DataFrame - to_positive is typically called
    on the *original* df afterwards, so the raw playtime values must still
    be intact."""
    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 1], "playtime": [0, 5, 100]})

    df_binary = PreprocessBlock().to_binary(df, playtime="playtime")

    assert df_binary["playtime"].tolist() == [0, 1, 1]
    assert df["playtime"].tolist() == [0, 5, 100]


def test_to_binary_raises_when_playtime_column_missing():
    df = pd.DataFrame({"user_id": [1], "item_id": [1]})

    with pytest.raises(AssertionError):
        PreprocessBlock().to_binary(df, playtime="playtime")


def test_to_positive_keeps_only_binarized_positive_rows():
    """to_positive's own assertion checks surviving values are exactly 1, not
    just > 0 - it's meant to run on an already-binarized column (see
    load_steam_data, which always calls to_binary first), not raw playtime."""
    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 1], "playtime": [0, 5, 100]})
    df_binary = PreprocessBlock().to_binary(df, playtime="playtime")

    df_positive = PreprocessBlock().to_positive(df_binary, playtime="playtime")

    assert df_positive["playtime"].tolist() == [1, 1]
    assert list(df_positive.index) == [1, 2]


def test_to_positive_raises_when_playtime_column_missing():
    df = pd.DataFrame({"user_id": [1], "item_id": [1]})

    with pytest.raises(AssertionError):
        PreprocessBlock().to_positive(df, playtime="playtime")


def _toy_train_df() -> pd.DataFrame:
    # 10 active users x 10 shared items - exactly the >=10/>=10 activity
    # threshold apply_filter enforces, so the filters' ">=" (inclusive)
    # semantics get exercised at the boundary.
    rows = [{"user_id": u, "item_id": i} for u in range(1, 11) for i in range(1, 11)]
    # An inactive user (too few items) - dropped by MinItemsPerUser.
    rows += [{"user_id": 99, "item_id": i} for i in (1, 2, 3)]
    # A rare item (too few users) - dropped by MinUsersPerItem.
    rows += [{"user_id": u, "item_id": 88} for u in (1, 2, 3)]
    return pd.DataFrame(rows)


def test_apply_filter_drops_inactive_users_and_rare_items_from_train():
    df_train = _toy_train_df()
    df_test = pd.DataFrame({"user_id": [1], "item_id": [1]})

    (train_im, _test_im), user_mapping, item_mapping = PreprocessBlock().apply_filter(
        df_train, df_test, user_col="user_id", item_col="item_id"
    )

    assert set(user_mapping.keys()) == set(range(1, 11))
    assert set(item_mapping.keys()) == set(range(1, 11))
    assert train_im.shape == (10, 10)
    assert train_im.values.nnz == 100


def test_apply_filter_restricts_test_set_to_users_and_items_kept_in_train():
    """A test-set row whose user or item got filtered out of train (i.e.
    never seen during training) must be dropped too - otherwise the
    resulting InteractionMatrix would carry ids outside train's id space."""
    df_train = _toy_train_df()
    df_test = pd.DataFrame(
        {
            "user_id": [1, 99, 1],  # user 99 was filtered out of train
            "item_id": [1, 1, 88],  # item 88 was filtered out of train
        }
    )

    (train_im, test_im), _user_mapping, _item_mapping = PreprocessBlock().apply_filter(
        df_train, df_test, user_col="user_id", item_col="item_id"
    )

    assert test_im.values.nnz == 1
    assert train_im.shape == test_im.shape
