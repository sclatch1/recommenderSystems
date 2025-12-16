"""
Clean RecPack BPRMF pipeline for MovieLens-1M.

Fixes applied:
 - Convert ratings -> implicit binary (rating > 3 -> 1)
 - Use explicit filters (deterministic)
 - Properly pass params to PipelineBuilder.add_algorithm(...)
 - Use sparse masking scores.multiply(...) to prevent recommending training/test items
 - Use save_best / keep_last correctly so best model is restored
"""

import argparse
from unittest import case
import numpy as np
import pandas as pd
import scipy as sp

from recpack.datasets import MovieLens1M, DummyDataset
from recpack.scenarios import WeakGeneralization, StrongGeneralization
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, MinRating, Filter
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.pipelines import PipelineBuilder
from src.my_bprmf import MyBPRMF, PPAC_BPRMF
from recpack.pipelines import ALGORITHM_REGISTRY
from recpack.algorithms import Popularity, ItemKNN

from recpack.scenarios.splitters import FractionInteractionSplitter




from src.preprocess_block import PreprocessBlock
from src.utils import scores2recommendations, save_metrics_incremental
from src.splitting_block import SplittingBlock
from src.new_metrics import ItemGiniK, 


from hyperopt import hp
from recpack.pipelines.hyperparameter_optimisation import HyperoptInfo

from src.metrics import calculate_ndcg, calculate_calibrated_recall, calculate_item_gini, calculate_publisher_gini


# Helper: convert sparse top-K rank matrix to dataframe (user,item,rank)
def matrix2df(X) -> pd.DataFrame:
    coo = sp.sparse.coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Only merged2 and dummy are supported in this script.")
    parser.add_argument("--algo", type=str, default="bprmf", help="Algorithm to use: 'bprmf' or 'itemknn'.")
    return parser.parse_args()


def hyperparam_tuning():
    optimisation_space = HyperoptInfo(
    {
        "num_components": hp.choice("num_components", [68, 16, 32, 64, 128, 256]),
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
        "lambda_h": hp.uniform("lambda_h", 0.001, 0.1),
        "lambda_w": hp.uniform("lambda_w", 0.001, 0.1),
        "max_epochs": hp.choice("max_epochs", [ 4, 8]),
        "batch_size": hp.choice("batch_size", [128, 256, 512])
    },
    max_evals=50,
    )

    return optimisation_space


def run_pipeline(args) -> pd.DataFrame:
    pb = PreprocessBlock()



    df_train = pd.concat([
        pd.read_csv("data/train_interactions.csv"), pd.read_csv("data/test_interactions_in.csv")
    ])

    df_test = pd.read_csv("data/test_interactions_in.csv")



    df_train = pb.to_binary(df_train, playtime="playtime")
    df_train = pb.to_positive(df_train, playtime="playtime")
    (train_interaction, test_interaction) , user_mapping, item_mapping = pb.apply_filter(df_train, df_test, user_col="user_id", item_col="item_id", is_test=False)



    print("Train interaction shape:", train_interaction.shape)
    print("Test interaction shape:", test_interaction.shape)

    # scenario = WeakGeneralization(0.7, validation=True, seed=42)
    # scenario.split(train_interaction)

    # scenario = StrongGeneralization(0.7, validation=True, seed=42)
    # scenario.split(train_interaction)

    # optimisation_space = hyperparam_tuning()
    # params = {
    #     "validation_sample_size": 200,
    #     "predict_topK": 20,
    #     "seed": 42,
    #     "save_best_to_file": False,
    #     "keep_last": True
    # }

    splitter = FractionInteractionSplitter(in_frac=0.7,    
        seed=42
    )


    X_train_fit, X_val = splitter.split(train_interaction)
    X_val = splitter.split(X_val)


    test = splitter.split(test_interaction)

    match args.algo:
        case "bprmf":
            params, hyper_params = bprmf(include_hyperparams=False)
        case "ppac":
            params, hyper_params = ppac(include_hyperparams=False)

    

    ALGORITHM_REGISTRY.register("PPAC_BPRMF", PPAC_BPRMF)
    ALGORITHM_REGISTRY.register("MyBPRMF", MyBPRMF)
    pb = PipelineBuilder()
    if args.algo == "ppac":
        pb.add_algorithm("PPAC_BPRMF", params=params, optimisation_info=hyper_params)
    else:
        pb.add_algorithm("MyBPRMF", params=params, optimisation_info=hyper_params)
    # pb.add_algorithm("ItemKNN", params={"K":200, "similarity":"cosine"})
    # pb.add_algorithm("EASE", params={"density": 0.8})
    # pb.add_algorithm("Popularity")
    # pb.set_data_from_scenario(scenario)
    pb.set_full_training_data(train_interaction)
    pb.set_validation_data(X_val)
    pb.set_validation_training_data(X_val[0])
    pb.set_test_data(test)
    if hyper_params != None:
        pb.set_optimisation_metric("NDCGK", 10)
    pb.add_metric("NDCGK", [10, 20, 30])
    # pb.add_metric("gini", [])
    pipe = pb.build()
    scores = pipe.run() 
    df_metrics = pipe.get_metrics()
    if hyper_params != None:
        param_info = pipe.optimisation_results
        print(param_info)
        for i, r in enumerate(param_info):
            print(r)

    print("Metrics:")
    print(df_metrics)

    save_metrics_incremental("metrics/", df_metrics, prefix=f"{args.algo}")


    # # model = MyBPRMF(**bpr_params)
    # # model.fit(X_train_fit, validation_data=X_val)
    # # scores = model.predict(test_interaction)
    

    df_recos = scores2recommendations(
                    scores,
                    test_interaction.binary_values,
                    recommendation_count=20,
                    user_id_mapping=user_mapping,
                    item_id_mapping=item_mapping,
                    prevent_history_recos=False
                )
    
    print(f"item_gini", calculate_item_gini(df_recos,k=20))

    df_recos.drop(columns=["rank"], inplace=True)

    return df_recos




def bprmf(include_hyperparams=True):
    """
    BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space to keep optimization time reasonable.

    Args:
        include_hyperparams (bool): 
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        dict or tuple: 
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: default training parameters (bpr_params)
    """

    if include_hyperparams:
        # ------------------------------------------------------------------
        # Fixed BPRMF parameters
        # ------------------------------------------------------------------
        fixed_params = {
            "max_epochs": 8,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }

        # ------------------------------------------------------------------
        # Hyperopt search space (only for tunable parameters)
        # ------------------------------------------------------------------
        hyperopt_space = HyperoptInfo(
            {
                "num_components": hp.choice(
                    "num_components",
                    [64, 96, 128]
                ),
                "learning_rate": hp.loguniform(
                    "learning_rate",
                    np.log(1e-4),
                    np.log(5e-3)
                ),
                "lambda_h": hp.loguniform(
                    "lambda_h",
                    np.log(1e-6),
                    np.log(1e-3)
                ),
                "lambda_w": hp.loguniform(
                    "lambda_w",
                    np.log(1e-6),
                    np.log(1e-3)
                ),
                "batch_size": hp.choice(
                    "batch_size",
                    [256, 512]
                ),
            },
            max_evals=1
        )

        return fixed_params, hyperopt_space

    else:
        # ------------------------------------------------------------------
        # Default parameters for training (no hyperparameter search)
        # ------------------------------------------------------------------
        bpr_params = {
            "num_components": 128,       # Reduce - easier to learn
            "learning_rate": 0.005,      # Higher LR
            "lambda_h": 0.0,           
            "lambda_w": 0.0,
            "max_epochs": 4,           
            "batch_size": 512,           # Larger batches
            "save_best_to_file": False,         
            "keep_last": False,         # Don't keep last model
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }
        return bpr_params, None



def ppac(include_hyperparams=True):
    """
    PPAC-BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space for reasonable optimization time.

    Args:
        include_hyperparams (bool): 
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        dict or tuple: 
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: default training parameters (ppac_params)
    """

    if include_hyperparams:
        # ------------------------------------------------------------------
        # Fixed parameters
        # ------------------------------------------------------------------
        fixed_params = {
            "max_epochs": 8,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }

        # ------------------------------------------------------------------
        # Hyperopt search space (tunable parameters)
        # ------------------------------------------------------------------
        hyperopt_space = HyperoptInfo(
            {
                "gamma": hp.uniform(
                    "gamma",
                    0.05,
                    0.3
                ),
                "beta": hp.uniform(
                    "beta",
                    0.05,
                    0.3
                ),
                "reg_coe": hp.loguniform(
                    "reg_coe",
                    np.log(1e-4),
                    np.log(1e-2)
                ),
                "l2_coe": hp.loguniform(
                    "l2_coe",
                    np.log(1e-5),
                    np.log(1e-3)
                ),
            },
            max_evals=1
        )

        return fixed_params, hyperopt_space

    else:
        # ------------------------------------------------------------------
        # Default parameters for training (no hyperparameter search)
        # ------------------------------------------------------------------
        ppac_params = {
            "gamma": 64,                 # default novelty weight
            "beta": -32,                  # default novelty weight             
            "reg_coe": 0.000001,
            "l2_coe":0.0000001,
            "max_epochs": 8,              
            "save_best_to_file": False,         
            "keep_last": False,            
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
            "num_components": 128,       # Reduce - easier to learn
            "learning_rate": 0.005,       # Higher LR
            "lambda_h": 0.0,           
            "lambda_w": 0.0,
            "max_epochs": 4,           
            "batch_size": 512,           # Larger batches
            "save_best_to_file": False,         
            "keep_last": False,         # Don't keep last model
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,

        }
        return ppac_params, None


    


def run_mostpop_pipeline():
    pb = PreprocessBlock()

    df_train = pd.read_csv("data/train_interactions.csv")
    df_train = pb.to_binary(df_train, "playtime")
    df_train = pb.to_positive(df_train, "playtime")

    
    
    train_interaction = pb.apply_filter(
        df_train, 
        user_col="user_id", 
        item_col="item_id", 
        is_test=False
    )

    df_test_in = pd.read_csv("data/test_interactions_in.csv")
    df_test_in = pb.to_binary(df_test_in, "playtime")
    df_test_in = pb.to_positive(df_test_in, "playtime")

    

    test_interaction = pb.apply_filter(
        df_test_in,
        user_col="user_id",
        item_col="item_id",
        is_test=True
    )

    model = Popularity()
    
    model.fit(train_interaction)
    
    scores = model.predict(test_interaction)

    df_recos = scores2recommendations(
        scores,
        test_interaction.binary_values,
        recommendation_count=20,
        prevent_history_recos=True
    )

    df_recos.drop(columns=["rank"], inplace=True)

    return df_recos

def main():
    args = parse_args()


    recommendation = run_pipeline(args)

    recommendation.to_csv(f"output/{args.algo}_recommendations.csv", index=False)




if __name__ == "__main__":
    main()
