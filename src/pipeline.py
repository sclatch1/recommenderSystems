import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from hyperopt import hp
from recpack.algorithms.base import TorchMLAlgorithm
from recpack.pipelines import ALGORITHM_REGISTRY, METRIC_REGISTRY, PipelineBuilder
from recpack.pipelines.hyperparameter_optimisation import HyperoptInfo

from src.algorithm import PPAC_BPRMF
from src.metrics import (
    PPRUK,
    PRUK,
    ItemCoverageK,
    ItemGiniK,
    MeanItemPopularityK,
    cal_local_nov,
    calculate_item_gini,
    calculate_mean_popularity,
)
from src.plotting import plot_metrics_comparison
from src.preprocessing import PreprocessBlock
from src.splitter import PPACEqualExposureSplitter
from src.utils import (
    add_pct_change_vs_baseline,
    aggregate_metrics_over_seeds,
    compute_long_tail_coverage,
    create_comprehensive_metrics_table,
    generate_analysis_report,
    log_interaction_stats,
    save_metrics_incremental,
    save_recommendations_incremental,
    scores2recommendations,
)

warnings.simplefilter("ignore", sp.sparse.SparseEfficiencyWarning)

MINUTE = 60
SEEDS = [4, 7, 12]

SPLITS = [
    ("equal_exposure", True),
    ("natural", False),
]


def bprmf(include_hyperparams: bool = True, epochs: int = 20) -> tuple[dict, HyperoptInfo | None]:
    """
    BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space to keep optimization time reasonable.

    Args:
        include_hyperparams (bool):
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        tuple[dict, HyperoptInfo | None]:
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: (default training parameters, None)
    """

    if include_hyperparams:
        fixed_params = {
            "max_epochs": epochs,
            "stopping_criterion": "bpr",
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": None,
            "predict_topK": 20,
        }

        hyperopt_space = HyperoptInfo(
            {
                "num_components": hp.choice("num_components", [64, 128, 256]),
                "learning_rate": hp.choice("learning_rate", [0.005, 0.01]),
                "batch_size": hp.choice("batch_size", [256, 512]),
                "lambda_h": hp.choice("lambda_h", [1e-5, 1e-4, 1e-3, 1e-2]),
                "lambda_w": hp.choice("lambda_w", [1e-5, 1e-4, 1e-3, 1e-2]),
            },
            timeout=MINUTE * 45,
        )

        return fixed_params, hyperopt_space

    else:
        bpr_params = {
            "num_components": 128,
            "learning_rate": 0.005,
            "lambda_h": 0.0000455332,
            "lambda_w": 0.0001938869,
            "max_epochs": epochs,
            "stopping_criterion": "bpr",
            "batch_size": 512,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }
        return bpr_params, None


def ppac(include_hyperparams: bool = True, epochs: int = 20) -> tuple[dict, HyperoptInfo | None]:
    """
    PPAC-BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space for reasonable optimization time.

    Args:
        include_hyperparams (bool):
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        tuple[dict, HyperoptInfo | None]:
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: (default training parameters, None)
    """

    if include_hyperparams:
        fixed_params = {
            "max_epochs": epochs,
            "stopping_criterion": "bpr",
            "batch_size": 256,
            "learning_rate": 0.001,
            "num_components": 256,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": None,
            "predict_topK": 100,
        }

        hyperopt_space = HyperoptInfo(
            {
                "gamma": hp.choice("gamma", [128, 256, 512]),
                "beta": hp.choice("beta", [-128, -512, -1024]),
                "l2_coe": hp.choice("l2_coe", [1e-4, 1e-3, 1e-2, 1e-1]),
                "reg_coe": hp.choice("reg_coe", [1e-3, 1e-2, 1e-1, 1.0]),
            },
            timeout=MINUTE * 45,
        )

        return fixed_params, hyperopt_space

    else:
        ppac_params = {
            "gamma": 256,
            "beta": -128,
            "reg_coe": 1e-3,
            "l2_coe": 1e-4,
            "num_components": 128,
            "learning_rate": 0.005,  # higher LR
            "max_epochs": epochs,
            "stopping_criterion": "bpr",
            "batch_size": 512,  #    larger batches
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }
        return ppac_params, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--analyze", action="store_true", help="Generate comprehensive analysis")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to run")
    return parser.parse_args()


def load_steam_data():
    """
    Load and preprocess the Steam interaction data.
    """
    pb_prep = PreprocessBlock()

    df_train = pd.concat([pd.read_csv("data/train_interactions.csv"), pd.read_csv("data/test_interactions_in.csv")])
    df_test = pd.read_csv("data/test_interactions_in.csv")

    df_train = pb_prep.to_binary(df_train, playtime="playtime")
    df_train = pb_prep.to_positive(df_train, playtime="playtime")
    (train_interaction, test_interaction), user_mapping, item_mapping = pb_prep.apply_filter(
        df_train, df_test, user_col="user_id", item_col="item_id", is_test=False
    )

    print("Train interaction shape:", train_interaction.shape)
    print("Test interaction shape:", test_interaction.shape)
    log_interaction_stats("Steam - before split (full preprocessed pool)", train_interaction)

    return train_interaction, test_interaction, df_train, df_test, user_mapping, item_mapping


def _register_algorithms_and_metrics(popularity_state: dict) -> None:
    """Register PPAC_BPRMF and the custom metrics once."""
    if "PPAC_BPRMF" not in ALGORITHM_REGISTRY:
        ALGORITHM_REGISTRY.register("PPAC_BPRMF", PPAC_BPRMF)

    if "MeanPopK" not in METRIC_REGISTRY:
        METRIC_REGISTRY.register("MeanPopK", lambda K: MeanItemPopularityK(K, popularity_state["item_popularity"]))

    if "gini_item" not in METRIC_REGISTRY:
        METRIC_REGISTRY.register("gini_item", ItemGiniK)

    if "item_coverage" not in METRIC_REGISTRY:
        METRIC_REGISTRY.register("item_coverage", ItemCoverageK)

    if "PRUK" not in METRIC_REGISTRY:
        METRIC_REGISTRY.register("PRUK", lambda K: PRUK(K, popularity_state["item_popularity"]))

    if "PPRUK" not in METRIC_REGISTRY:
        METRIC_REGISTRY.register("PPRUK", lambda K: PPRUK(K, popularity_state["local_popularity"]))


def _run_pipeline_for_seed(
    args,
    seed: int,
    split_name: str,
    resample_test: bool,
    train_interaction,
    test_interaction,
    df_train,
    df_test,
    user_mapping,
    item_mapping,
    popularity_state,
) -> pd.DataFrame:
    """Split, train, and evaluate once for a single (seed, split protocol)
    pair. Returns one row per algorithm (short names), tagged with this seed
    and split, for aggregation across seeds/splits by `run_pipeline`."""

    splitter = PPACEqualExposureSplitter(
        test_frac=0.1, validation_frac=0.1, seed=seed, resample_validation=False, resample_test=resample_test
    )
    train_split, val_split, test_split = splitter.split(train_interaction)
    log_interaction_stats(f"Steam - after split ({split_name}, seed={seed}): train", train_split)
    log_interaction_stats(f"Steam - after split ({split_name}, seed={seed}): val", val_split)
    log_interaction_stats(f"Steam - after split ({split_name}, seed={seed}): test", test_split)

    params, hyper_params = bprmf(include_hyperparams=args.optimize, epochs=args.epochs)
    params_p, hyper_params_p = ppac(include_hyperparams=args.optimize, epochs=args.epochs)

    # Reseed model init/negative sampling to match this pass's split seed -
    # bprmf()/ppac()'s own "seed": 42 default is just a fallback for
    # standalone use of those functions.
    params["seed"] = seed
    params_p["seed"] = seed

    popularity_state["item_popularity"] = np.array(train_split.values.sum(axis=0)).ravel()
    popularity_state["local_popularity"] = cal_local_nov(train_split.values)[1].numpy()

    pb = PipelineBuilder("results")
    pb.add_algorithm("PPAC_BPRMF", params=params_p, optimisation_info=hyper_params_p)
    pb.add_algorithm("BPRMF", params=params, optimisation_info=hyper_params)
    pb.set_full_training_data(train_split)
    pb.set_validation_training_data(train_split)
    pb.set_validation_data((train_split, val_split))
    pb.set_test_data((train_split, test_split))

    if args.optimize:
        pb.set_optimisation_metric("NDCGK", 10)

    pb.add_metric("NDCGK", [10, 20])
    pb.add_metric("gini_item", [10, 20])
    pb.add_metric("item_coverage", [10, 20])
    pb.add_metric("RecallK", [10, 20])
    pb.add_metric("MeanPopK", [10, 20])
    pb.add_metric("PRUK", [10, 20])
    pb.add_metric("PPRUK", [10, 20])

    pipe = pb.build()
    pipe.run()
    df_metrics = pipe.get_metrics()
    df_optimization = pipe.optimisation_results if args.optimize else None
    pipe.save_metrics()

    print(f"\nMetrics (Steam, {split_name} split, seed={seed}):")
    print(df_metrics)

    df_metrics_named = pipe.get_metrics(short=True).reset_index(names="algorithm")
    df_pct = add_pct_change_vs_baseline(df_metrics_named, baseline="BPRMF")
    print("\nPercent change vs BPRMF baseline:")
    print(df_pct)

    save_metrics_incremental("metrics/", df_metrics, df_optimization, prefix=f"{split_name}_seed{seed}")

    if args.analyze:
        _run_analysis(
            pipe,
            algorithm_configs=[
                ("PPAC_BPRMF", params_p),
                ("BPRMF", params),
            ],
            df_train=df_train,
            df_test=df_test,
            test_interaction=test_interaction,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )

    df_metrics_named["seed"] = seed
    df_metrics_named["split"] = split_name
    return df_metrics_named


def run_pipeline(args) -> pd.DataFrame:
    """Run the Steam pipeline over every seed in `SEEDS` and every protocol
    in `SPLITS`.
    Returns a DataFrame of metrics aggregated over seeds and splits, with one
    row per algorithm (short name) and columns for mean and std of each metric.
    """
    seeds = SEEDS
    splits = SPLITS

    if (len(seeds) > 1 or len(splits) > 1) and (args.optimize or args.analyze):
        raise ValueError("--optimize/--analyze aren't supported together with multiple SEEDS or multiple SPLITS. ")

    train_interaction, test_interaction, df_train, df_test, user_mapping, item_mapping = load_steam_data()

    popularity_state: dict = {}
    _register_algorithms_and_metrics(popularity_state)

    per_run_metrics = [
        _run_pipeline_for_seed(
            args,
            seed,
            split_name,
            resample_test,
            train_interaction,
            test_interaction,
            df_train,
            df_test,
            user_mapping,
            item_mapping,
            popularity_state,
        )
        for split_name, resample_test in splits
        for seed in seeds
    ]
    df_metrics_all_runs = pd.concat(per_run_metrics, ignore_index=True)

    if len(seeds) == 1:
        return df_metrics_all_runs

    df_summary = aggregate_metrics_over_seeds(df_metrics_all_runs, group_cols=["algorithm", "split"])
    print(f"\nMetrics aggregated over {len(seeds)} seeds {seeds} x {len(splits)} splits (mean +/- std):")
    print(df_summary)

    save_metrics_incremental("metrics/", df_summary, None, prefix="multiseed_summary")

    return df_summary


def _run_analysis(
    pipe,
    algorithm_configs,
    df_train,
    df_test,
    test_interaction,
    user_mapping,
    item_mapping,
):
    """
    Re-trains each pipeline algorithm outside the PipelineBuilder (which
    doesn't expose per-algorithm prediction scores after pipe.run()) to get
    score matrices for recommendation export and comparative analysis
    plots/report.
    """
    # pipe.get_metrics() indexes by the algorithm's full-param repr, not a
    # column - the plotting/report helpers below all expect a plain
    # "algorithm" column, so build that via recpack's own short=True naming
    # (it splits the repr on "(", e.g. "PPAC_BPRMF(...)" -> "PPAC_BPRMF").
    df_metrics = pipe.get_metrics(short=True).reset_index(names="algorithm")

    scores = []
    for algo_name, algo_params in algorithm_configs:
        algorithm = ALGORITHM_REGISTRY.get(algo_name)(**algo_params)
        if isinstance(algorithm, TorchMLAlgorithm):
            pipe._train(algorithm, pipe.validation_training_data)
        else:
            pipe._train(algorithm, pipe.full_training_data)
        X_pred = pipe._predict_and_postprocess(algorithm, pipe.test_data_in)
        scores.append((algo_name, X_pred))

    # Total catalog size (not just the items that end up recommended) -
    # calculate_item_gini needs this so items with zero recommendations
    # still count toward the distribution instead of being dropped.
    total_items = df_train["item_id"].nunique()

    all_recommendations = {}
    for algorithm, score in scores:
        df_recos = scores2recommendations(
            score,
            test_interaction.binary_values,
            recommendation_count=10,
            user_id_mapping=user_mapping,
            item_id_mapping=item_mapping,
            prevent_history_recos=True,
        )

        print(f"item_gini_{algorithm}", calculate_item_gini(df_recos, k=10, total_items=total_items))
        print(f"mean_popularity_{algorithm}", calculate_mean_popularity(df_recos, df_train, 10))

        all_recommendations[algorithm] = df_recos
        df_recos = df_recos.drop(columns=["rank"])

        save_recommendations_incremental("output", df_recos, algorithm=algorithm)

    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating comprehensive analysis...")
    comprehensive_metrics = create_comprehensive_metrics_table(
        all_recommendations, df_train, df_test, k=10, metrics_df=df_metrics
    )
    print(comprehensive_metrics)
    comprehensive_metrics.to_csv(output_dir / "comprehensive_metrics.csv", index=False)

    plot_metrics_comparison(df_metrics, output_dir)

    popularity_dist = compute_long_tail_coverage(all_recommendations, df_train)
    popularity_dist.to_csv(output_dir / "popularity_distribution.csv", index=False)

    generate_analysis_report(df_metrics, comprehensive_metrics, popularity_dist, output_dir)

    print(f"Analysis outputs saved to: {output_dir}/")


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
