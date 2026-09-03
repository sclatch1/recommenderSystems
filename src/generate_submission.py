import argparse
from pathlib import Path

import pandas as pd

from src.algorithm import PPAC_BPRMF
from src.pipeline import load_steam_data, ppac
from src.splitter import PPACEqualExposureSplitter
from src.utils import scores2recommendations

SEED = 42


def generate_submission(k: int = 20, output_path: str = "submission.csv") -> pd.DataFrame:
    train_interaction, test_interaction, _, df_test, user_mapping, item_mapping = load_steam_data()

    splitter = PPACEqualExposureSplitter(test_frac=0.0, validation_frac=0.1, seed=SEED, resample_validation=False)
    train_split, val_split, _ = splitter.split(train_interaction)

    params, _ = ppac(include_hyperparams=False, epochs=20)
    params["seed"] = SEED

    algorithm = PPAC_BPRMF(**params)
    algorithm.fit(train_split, (train_split, val_split))

    X_pred = algorithm.predict(test_interaction)
    df_recos = scores2recommendations(
        X_pred,
        test_interaction.binary_values,
        recommendation_count=k,
        user_id_mapping=user_mapping,
        item_id_mapping=item_mapping,
        prevent_history_recos=True,
    )
    df_recos = df_recos.drop(columns=["rank"])

    real_test_users = set(df_test["user_id"].unique()) & set(user_mapping.keys())
    df_recos = df_recos[df_recos["user_id"].isin(real_test_users)]

    all_test_users = set(df_test["user_id"].unique())
    missing_users = all_test_users - real_test_users
    if missing_users:
        print(
            f"{len(missing_users)} of {len(all_test_users)} test users had too few interactions "
            "to get a trained embedding and are not covered by this submission (no fallback)."
        )

    df_recos = df_recos.sort_values(["user_id", "item_id"]).reset_index(drop=True)

    output_path = Path(output_path)
    df_recos.to_csv(output_path, index=False)
    print(f"Saved {len(df_recos)} recommendations for {df_recos['user_id'].nunique()} users to {output_path}")

    return df_recos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Number of recommended items per user")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_submission(k=args.k, output_path=args.output)


if __name__ == "__main__":
    main()
