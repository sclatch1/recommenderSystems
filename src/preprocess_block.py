import pandas as pd
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
import numpy as np
from recpack.pipelines import PipelineBuilder
from recpack.pipelines import ALGORITHM_REGISTRY

from src.utils import scores2recommendations
from src.my_bprmf import MyBPRMF
from src.splitting_block import SplittingBlock
from recpack.scenarios import WeakGeneralization

class PreprocessBlock:
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

    def apply_filter(self, df_train: pd.DataFrame, df_test: pd.DataFrame ,user_col: str, item_col: str, is_test: bool = False) -> InteractionMatrix:
        """
        Convert a pandas DataFrame to an InteractionMatrix.
        
        :param df: DataFrame with user, item columns
        :param user_col: Column name for user IDs
        :param item_col: Column name for item IDs
        :param is_test: If True, use the existing preprocessor fitted on training data
        :return: InteractionMatrix
        """
        # print("shape of df before processing:", df_train.shape)
        # if is_test:
        #     # Use the existing preprocessor that was fitted on training data
            
        #     assert self.preprocessor is not None, "Must process training data first"
        #     interaction_matrix = self.preprocessor.process(df_test)
            
        # else:
            # Create and fit preprocessor on training data
        self.preprocessor = DataFramePreprocessor(
            user_ix=user_col,
            item_ix=item_col,
        )
        
        # Add filters only for training data
        # self.preprocessor.add_filter(MinItemsPerUser(min_items_per_user=10, item_ix=item_col, user_ix=user_col))
        # self.preprocessor.add_filter(MinUsersPerItem(min_users_per_item=10, item_ix=item_col, user_ix=user_col))
        
        interaction_matrices = self.preprocessor.process_many(df_train, df_test)

        self.user_id_mapping = self.preprocessor._user_id_mapping
        self.item_id_mapping = self.preprocessor._item_id_mapping
        
        for im in interaction_matrices:
            assert isinstance(im, InteractionMatrix), "Conversion to InteractionMatrix failed"
        return interaction_matrices, self.user_id_mapping, self.item_id_mapping
    



if __name__ == "__main__":
    pb = PreprocessBlock()
    
    df_merged = pd.concat([
        pd.read_csv("data/train_interactions.csv"),
        pd.read_csv("data/test_interactions_in.csv")
    ])


    # Process TRAINING data
    df_train = pd.read_csv("data/train_interactions.csv")
    df_train = pb.to_binary(df_train, playtime="playtime")
    df_train = pb.to_positive(df_train, playtime="playtime")
    train_interaction = pb.apply_filter(df_train, user_col="user_id", item_col="item_id", is_test=False)
    
    # Process TEST data (using same mappings)
    df_test = pd.read_csv("data/test_interactions_in.csv")
    df_test = pb.to_binary(df_test, playtime="playtime")
    df_test = pb.to_positive(df_test, playtime="playtime")
    test_interaction = pb.apply_filter(df_test, user_col="user_id", item_col="item_id", is_test=True)

    df_merged = pb.to_binary(df_merged, playtime="playtime")
    df_merged = pb.to_positive(df_merged, playtime="playtime")
    merged_interaction = pb.apply_filter(df_merged, user_col="user_id", item_col="item_id", is_test=False)

    scenario = WeakGeneralization(0.7, validation=True, seed=42)
    scenario.split(merged_interaction)

    # Now split training data for validation
    sb = SplittingBlock(train_interaction, test_interaction)


    print("density:", train_interaction.density)
    print("average user interactions:", train_interaction.num_interactions / train_interaction.shape[0])

    bpr_params = {
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

    ALGORITHM_REGISTRY.register(MyBPRMF.__name__, MyBPRMF)

    print(sb.get_test_sets()[0].shape)
    print(sb.get_test_sets()[1].shape)
    print(sb.get_validation_sets()[0].shape)
    print(sb.get_validation_sets()[1].shape)
    print(sb.get_full_training_data().shape)

    # Build pipeline and pass BPRMF parameters through PipelineBuilder
    pb = PipelineBuilder()
    # pb.add_algorithm("MyBPRMF", params=bpr_params)
    pb.add_algorithm("ItemKNN", params={"K": 200, "similarity": "cosine"})  
    # pb.set_full_training_data(sb.get_full_training_data())
    # pb.set_test_data(sb.get_test_sets())
    # pb.set_validation_data(sb.get_validation_sets())
    # pb.set_validation_training_data(sb.get_validation_training_data())
    #pb.set_optimisation_metric("NDCGK", 10)
    pb.set_data_from_scenario(scenario)
    pb.add_metric("NDCGK", 10)
    pipe = pb.build()
    scores = pipe.run()

    df_metrics = pipe.get_metrics()
    print("\nFinal Model Metrics (from pipeline):")
    print(f"the metrics are {df_metrics.head(20)}")
    df_metrics.to_csv("bprmf_metrics.csv", index=False)



    df_recos = scores2recommendations(scores,
                                        X_test_in=test_interaction,
                                        recommendation_count=20,prevent_history_recos=False)
    
    df_recos.drop(columns=["rank"], inplace=True)

    df_recos.to_csv("bprmf_recommendations.csv", index=False)