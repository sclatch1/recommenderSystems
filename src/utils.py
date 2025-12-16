import pandas as pd
import scipy as sp
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix as Inter
from pathlib import Path

# Helper: convert sparse top-K rank matrix to dataframe (user,item,rank)
def matrix2df(X) -> pd.DataFrame:
    coo = sp.sparse.coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })




def scores2recommendations(
    scores: sp.sparse.csr_matrix, 
    X_test_in: sp.sparse.csr_matrix, 
    recommendation_count: int,
    user_id_mapping: dict,
    item_id_mapping: dict,
    prevent_history_recos = True
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
    df_recos['user_id'] = df_recos['user_id'].map(reverse_user_mapping)
    df_recos['item_id'] = df_recos['item_id'].map(reverse_item_mapping)
    
    df_recos = df_recos.sort_values(["user_id", "rank"])
    
    return df_recos


def find_common(df1, df2):
    users1 = set(df1['user_id'])
    users2 = set(df2['user_id'])
    common_users = users1.intersection(users2)
    percentage_df1 = (len(common_users) / len(users1)) * 100
    print(f"{percentage_df1:.2f}% of df1 users also appear in df2")
    percentage_df2 = (len(common_users) / len(users2)) * 100
    print(f"{percentage_df2:.2f}% of df2 users also appear in df1")


def save_metrics_incremental(dir_path, df_metrics, prefix="test", suffix=".csv"):
    """
    Save df_metrics to the next available file name like:
    test_0.csv, test_1.csv, test_2.csv, ...

    Args:
        dir_path (str or Path): Directory where the file should be saved.
        df_metrics (pd.DataFrame): Metrics DataFrame to save.
        prefix (str): File prefix (default: "test")
        suffix (str): File suffix (default: ".csv")
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Find existing files matching prefix_<n>.suffix
    existing_indices = []
    for path in dir_path.glob(f"{prefix}_*{suffix}"):
        try:
            idx = int(path.stem.split("_")[-1])
            existing_indices.append(idx)
        except ValueError:
            continue

    next_idx = max(existing_indices) + 1 if existing_indices else 0
    output_path = dir_path / f"{prefix}_{next_idx}{suffix}"

    df_metrics.to_csv(output_path)
    return output_path


if "__main__" == __name__:
    df1 = pd.read_csv("output/bprmf_recommendations.csv")
    df2 = pd.read_csv("data/test_interactions_in.csv")

    find_common(df1=df1,df2=df2)
