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


from pathlib import Path
import pandas as pd

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


def print_model_config_as_dict(model_str):
    """
    Convert a model configuration string to a dict.
    If input is not a string, return it unchanged.
    """
    if not isinstance(model_str, str):
        return model_str

    if "(" not in model_str or ")" not in model_str:
        return model_str

    params_str = model_str[model_str.find("(") + 1 : model_str.rfind(")")]
    params = [p.strip() for p in params_str.split(",")]

    config = {}
    for p in params:
        key, value = p.split("=", 1)

        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        config[key] = value

    return config


import pandas as pd
import numpy as np

def debiasing_analysis(y_true, y_pred_topK, n_groups=5):
    """
    Analyze debiasing effect by item popularity groups.

    Parameters
    ----------
    y_true : csr_matrix or DataFrame
        True user-item interactions (binary).
        Shape: n_users x n_items
    y_pred_topK : csr_matrix or DataFrame
        Top-K recommended items (binary, 1 if recommended).
        Shape: n_users x n_items
    n_groups : int
        Number of popularity groups (head → tail)
    
    Returns
    -------
    group_stats : pd.DataFrame
        DataFrame with columns:
        - group: group index (0=head, n_groups-1=tail)
        - item_proportion: fraction of items in this group
        - rec_frequency: fraction of recommendations going to this group
        - recall: recall of items in this group
    """
    if not isinstance(y_true, pd.DataFrame):
        y_true = pd.DataFrame(y_true.toarray())
    if not isinstance(y_pred_topK, pd.DataFrame):
        y_pred_topK = pd.DataFrame(y_pred_topK.toarray())
    
    n_items = y_true.shape[1]

    # 1️⃣ Compute item popularity in training set
    item_popularity = y_true.sum(axis=0)  # number of interactions per item
    item_ranks = np.argsort(-item_popularity)  # descending popularity

    # 2️⃣ Divide items into n_groups
    items_per_group = n_items // n_groups
    item_group = np.zeros(n_items, dtype=int)
    for g in range(n_groups):
        start = g * items_per_group
        end = (g + 1) * items_per_group if g < n_groups - 1 else n_items
        item_group[item_ranks[start:end]] = g

    # 3️⃣ Compute recommendation frequency per group
    rec_counts = y_pred_topK.sum(axis=0).values.ravel()  # sum over users
    rec_frequency = []
    group_item_proportion = []
    recall_per_group = []

    for g in range(n_groups):
        group_items = np.where(item_group == g)[0]
        group_item_proportion.append(len(group_items) / n_items)

        # Recommendations in this group
        rec_frequency.append(rec_counts[group_items].sum() / rec_counts.sum())

        # Recall for this group
        relevant = y_true.iloc[:, group_items].values
        recommended = y_pred_topK.iloc[:, group_items].values

        recall = ((relevant > 0) & (recommended > 0)).sum() / (relevant > 0).sum() if (relevant > 0).sum() > 0 else np.nan
        recall_per_group.append(recall)


    # 4️⃣ Build summary DataFrame
    group_stats = pd.DataFrame({
        "group": range(n_groups),
        "item_proportion": group_item_proportion,
        "rec_frequency": rec_frequency,
        "recall": recall_per_group
    })

    return group_stats



def plot(group_stats):
    import matplotlib.pyplot as plt

    plt.bar(group_stats["group"], group_stats["item_proportion"], alpha=0.5, label="Item Proportion")
    plt.plot(group_stats["group"], group_stats["rec_frequency"], marker='o', label="Rec Frequency")
    plt.plot(group_stats["group"], group_stats["recall"], marker='x', label="Recall")
    plt.xlabel("Popularity Group (0=head, N=tail)")
    plt.ylabel("Fraction / Recall")
    plt.legend()
    plt.show()




if "__main__" == __name__:
    df1 = pd.read_csv("output/ppac_recommendations.csv")
    df2 = pd.read_csv("data/test_interactions_in.csv")

    find_common(df1=df1,df2=df2)
