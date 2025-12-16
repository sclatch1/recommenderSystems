from recpack.scenarios.splitters import FractionInteractionSplitter, Splitter
from recpack.scenarios.scenario_base import Scenario
from recpack.matrix import InteractionMatrix
from recpack.pipelines import PipelineBuilder
from recpack.pipelines import ALGORITHM_REGISTRY

import pandas as pd
from src.my_bprmf import MyBPRMF

class SplittingBlock():
    def __init__(self, train: InteractionMatrix, test: InteractionMatrix, 
                 frac_data_in: float = 0.8, random_seed: int = 42):
        """
        Initializes the SplittingBlock with training and testing data,
        following the WeakGeneralization splitting strategy.

        Args:
            train (InteractionMatrix): InteractionMatrix containing training interactions.
            test (InteractionMatrix): InteractionMatrix containing testing interactions.
            frac_data_in (float): Fraction of data to use for training at each split level.
            random_seed (int): Random seed for reproducibility.
        """
        self.frac_data_in = frac_data_in
        self.random_seed = random_seed
        
        # Initialize the splitter
        self.splitter = FractionInteractionSplitter(
            in_frac=frac_data_in,
            seed=random_seed
        )
        
        # Full training data (IT_u) - this is your original train set
        self.full_training_data = train
        
        # Test data sets (already provided)
        self.test_data_in = train  # Same as full_training_data
        self.test_data_out = test  # Your test set
        
        # Split the full training data to create validation sets
        self.validation_training_data, self.validation_data_out = self.splitter.split(self.full_training_data)
        
        # validation_data_in is the same as validation_training_data
        self.validation_data_in = self.validation_training_data
    
    def get_validation_sets(self):
        """Returns validation training, in, and out datasets."""
        return self.validation_data_in, self.validation_data_out
    
    def get_test_sets(self):
        """Returns test in and out datasets."""
        return self.test_data_in, self.test_data_out
    
    def get_full_training_data(self):
        """Returns the full training dataset."""
        return self.full_training_data
    
    def get_validation_training_data(self):
        """Returns the validation training dataset."""
        return self.validation_training_data



# Usage example
if __name__ == "__main__":    
 

    sb = SplittingBlock()

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

    # Build pipeline and pass BPRMF parameters through PipelineBuilder
    pb = PipelineBuilder()
    # pb.add_algorithm("MyBPRMF", params=bpr_params)  
    pb.add_algorithm("BPRMF", params=bpr_params)
    pb.set_full_training_data(splitter.full_training_data)
    pb.set_test_data((test_in, test_out))
    pb.set_validation_data((splitter.validation_data_in, splitter.validation_data_out))
    pb.set_validation_training_data(splitter.validation_training_data)
    #pb.set_validation_data(sb.validation_data_in)
    #pb.set_optimisation_metric("NDCGK", 10)  
    pb.add_metric("NDCGK", 10)            

    pipe = pb.build()
    scores = pipe.run()

