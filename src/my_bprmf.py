import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from recpack.algorithms.bprmf import BPRMF
from recpack.algorithms.samplers import BootstrapSampler
from src.metrics import cal_local_nov_simple, cal_global_nov
from recpack.scenarios import WeakGeneralization
from recpack.datasets import DummyDataset
from hyperopt import hp
from recpack.pipelines import PipelineBuilder, HyperoptInfo
from recpack.matrix import InteractionMatrix
import pandas as pd
from recpack.pipelines import ALGORITHM_REGISTRY
from scipy.sparse import lil_matrix
from recpack.algorithms.bprmf import MFModule
import torch.optim as optim


class MyBPRMF(BPRMF):
    def __init__(
        self,
        num_components: int = 100,
        lambda_h: float = 0.0,
        lambda_w: float = 0.0,
        batch_size: int = 1_000,
        max_epochs: int = 20,
        learning_rate: float = 0.01,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: int = None,
        save_best_to_file: bool = False,
        sample_size=None,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            num_components=num_components,
            lambda_h=lambda_h,
            lambda_w=lambda_w,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            sample_size=sample_size,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self.num_components = num_components
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w

        self.sample_size = sample_size

        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size,
        )

    def _init_model(self, X):
        num_users, num_items = X.shape
        self.model_ = MFModule(num_users, num_items, num_components=self.num_components).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)


class PPAC_BPRMF(MyBPRMF):
    """
    Personalized Popularity-Aware Collaborative BPR-MF (PPAC-BPRMF)
    
    Extends BPR-MF with popularity-aware predictions using both local (user-specific)
    and global novelty patterns, following the PPAC framework.
    
    The model learns to predict item novelty/popularity at both global and local levels,
    and uses these predictions to adjust recommendation scores.
    
    Parameters
    ----------
    num_components : int, optional
        Dimensionality of the user and item embeddings, by default 128
    batch_size : int, optional
        Batch size for training, by default 1000
    max_epochs : int, optional
        Maximum number of training epochs, by default 10
    learning_rate : float, optional
        Learning rate for optimization, by default 0.001
    gamma : float, optional
        Weight for local novelty term, by default 0.1
    beta : float, optional
        Weight for global novelty term, by default 0.1
    reg_coe : float, optional
        Regularization coefficient for novelty predictors, by default 0.001
    l2_coe : float, optional
        L2 regularization coefficient for embeddings, by default 1e-4
    use_simplified_local : bool, optional
        Use simplified local novelty calculation (without similar users), by default True
    """
    
    def __init__(
        self,
        num_components= 128,
        batch_size=512,
        max_epochs=8,
        learning_rate=0.005,
        gamma=0.1,
        beta=0.1,
        reg_coe=0.001,
        l2_coe=1e-4,
        use_simplified_local=True,
        save_best_to_file=False,
        keep_last=False,
        predict_topK=100,
        validation_sample_size=200,
        seed=None,
        **kwargs
    ):
        super().__init__(
            num_components=num_components,
            batch_size=batch_size,
            lambda_h=0.0,
            lambda_w=0.0,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        
        self.gamma = gamma
        self.beta = beta
        self.reg_coe = reg_coe
        self.l2_coe = l2_coe
        self.use_simplified_local = use_simplified_local
        
        # These will be initialized in _init_model
        self.local_pred = None
        self.global_pred = None
        self.global_pop = None
        self.local_pop = None
        self.reg_loss_fn = nn.MSELoss()
        self.f = nn.Sigmoid()

        self.losses_ = None

    def _convert_sparse_to_train_records(self, X):
        """
        Convert sparse matrix to train_records dictionary format
        
        Parameters
        ----------
        X : scipy.sparse matrix
            User-item interaction matrix
        
        Returns
        -------
        dict
            Dictionary mapping user indices to lists of item indices
        """
        train_records = {}
        X_coo = X.tocoo()
        
        for user_idx, item_idx in zip(X_coo.row, X_coo.col):
            if user_idx not in train_records:
                train_records[user_idx] = []
            train_records[user_idx].append(item_idx)
        
        return train_records
    
    def _init_model(self, X):
        """Initialize model with novelty-aware components"""
        super()._init_model(X)
        
        # Initialize sampler
        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size
        )
        
        # Convert sparse matrix to train_records format
        num_users, num_items = X.shape
        train_records = self._convert_sparse_to_train_records(X)
        
        # Calculate global novelty using the original function
        global_nov, global_pop_counts = cal_global_nov(train_records, num_items)
        self.global_pop = global_nov.to(self.device)
        self.global_pop = F.normalize(self.global_pop.float(), dim=0)
        
        # Calculate local novelty
        if self.use_simplified_local:
            local_nov, local_pop_counts = cal_local_nov_simple(
                train_records, num_users, num_items
            )
        else:
            # If you have similar users, use the original cal_local_nov function
            # local_nov, local_pop_counts = cal_local_nov(dataset, sim_users, train_records, num_items)
            # For now, fall back to simplified version
            local_nov, local_pop_counts = cal_local_nov_simple(
                train_records, num_users, num_items
            )
        
        self.local_pop = local_nov.to(self.device)
        self.local_pop = F.normalize(self.local_pop.float(), dim=1)
        
        # Initialize novelty prediction networks
        latent_dim = self.num_components
        
        # Local novelty predictor (user-item interaction patterns)
        self.local_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, latent_dim)
        ).to(self.device)
        
        # Global novelty predictor (overall item novelty)
        self.global_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        ).to(self.device)
        
        # Add predictors to optimizer
        predictor_params = list(self.local_pred.parameters()) + list(self.global_pred.parameters())
        self.optimizer.add_param_group({'params': predictor_params})
    
    def _compute_loss(self, users, pos_items, neg_items):
        """
        Compute PPAC-BPR loss with novelty-aware regularization
        
        Parameters
        ----------
        users : torch.Tensor
            User indices
        pos_items : torch.Tensor
            Positive item indices
        neg_items : torch.Tensor
            Negative item indices
        
        Returns
        -------
        torch.Tensor
            Total loss value
        """
        # Get embeddings
        user_emb = self.model_.user_embedding_(users)
        pos_item_emb = self.model_.item_embedding_(pos_items)
        neg_item_emb = self.model_.item_embedding_(neg_items)
        
        # Compute base collaborative filtering scores
        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        
        # Predict local novelty patterns
        usr_ci_emb = self.local_pred(user_emb)
        pos_ci_emb = self.local_pred(pos_item_emb)
        neg_ci_emb = self.local_pred(neg_item_emb)
        
        pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
        neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))
        
        # Predict global novelty
        pos_ci_global = self.global_pred(pos_item_emb)
        neg_ci_global = self.global_pred(neg_item_emb)
        
        # Modulate scores with predicted novelty
        pos_scores = pos_scores * (pos_ci_local * pos_ci_global)
        neg_scores = neg_scores * (neg_ci_local * neg_ci_global)
        
        # BPR loss
        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # Regularization: align predictions with actual novelty
        local_reg_loss = (
            self.reg_loss_fn(
                pos_ci_local,
                self.local_pop[users.cpu(), pos_items.cpu()].to(self.device)
            ) +
            self.reg_loss_fn(
                neg_ci_local,
                self.local_pop[users.cpu(), neg_items.cpu()].to(self.device)
            )
        ) / 2
        
        global_reg_loss = (
            self.reg_loss_fn(pos_ci_global, self.global_pop[pos_items.cpu()].to(self.device)) +
            self.reg_loss_fn(neg_ci_global, self.global_pop[neg_items.cpu()].to(self.device))
        ) / 2
        
        reg_loss = local_reg_loss + global_reg_loss
        
        # L2 regularization on embeddings
        l2_loss = (
            torch.mean(torch.sum(torch.pow(user_emb, 2), dim=1) / 2.) +
            torch.mean(torch.sum(torch.pow(pos_item_emb, 2), dim=1) / 2.) +
            torch.mean(torch.sum(torch.pow(neg_item_emb, 2), dim=1) / 2.)
        )
        
        # Total loss (matching original PPAC formula)
        total_loss = cf_loss + self.reg_coe * reg_loss + self.l2_coe * l2_loss
        
        return total_loss
    
    def _train_epoch(self, X):
        """Train for one epoch with PPAC loss"""
        losses = []
        self.model_.train()
        self.local_pred.train()
        self.global_pred.train()
        
        for users, target_items, neg_items in self.sampler.sample(X, sample_size=self.sample_size):
            users = users.to(self.device)
            target_items = target_items.to(self.device)
            neg_items = neg_items.squeeze(-1).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute PPAC loss
            loss = self._compute_loss(users, target_items, neg_items)
            
            loss.backward()
            losses.append(loss.item())

            self.optimizer.step()
            

        self.losses_ = losses

        return np.mean(losses)
    
    def _batch_predict(self, X, users):
        """
        Predict scores with novelty-aware adjustments
        
        Parameters
        ----------
        X : scipy.sparse matrix
            User-item interaction matrix
        users : np.ndarray
            User indices to predict for
        
        Returns
        -------
        np.ndarray
            Predicted scores
        """
        self.model_.eval()
        self.local_pred.eval()
        self.global_pred.eval()
        with torch.no_grad():
            user_tensor = torch.tensor(users, device=self.device, dtype=torch.long)
            
            # Get embeddings
            user_emb = self.model_.user_embedding_(user_tensor)
            item_emb = self.model_.item_embedding_.weight
            
            # Base collaborative filtering scores
            scores = torch.matmul(user_emb, item_emb.T)
            
            # Predict novelty patterns
            user_ci_emb = self.local_pred(user_emb)
            item_ci_emb = self.local_pred(item_emb)
            
            pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
            pred_global = self.global_pred(item_emb).expand(scores.shape)
            
            # Get real novelty values
            real_local = self.local_pop[user_tensor]
            real_global = self.global_pop.expand(scores.shape)
            
            # Combine CF scores with novelty awareness (matching original PPAC formula)
            scores = (
                scores * (pred_local * pred_global) +
                self.gamma * real_local +
                self.beta * real_global
            )

            result = lil_matrix(X.shape)
            result[users] = scores.cpu().numpy()
            
            return result.tocsr()
    




if __name__ == "__main__":
    optimisation_info = HyperoptInfo(
    hp.choice(
        'bprmf', [
            {
                'num_components': 100,
                "learning_rate" : 0.01,
                "lambda_h": 0.001,
                "lambda_w": 0.001,
                "validation_sample_size": 100,
                "predict_topK": 100,
                "save_best_to_file": True,
                "keep_last": True,
                "batch_size": 256,
            }
        ]
    ),
    max_evals=50,
    )

    ppb = PPAC_BPRMF(num_components=100, learning_rate=0.01, lambda_h=0.001, lambda_w=0.001, validation_sample_size=100,predict_topK=100, save_best_to_file=True, keep_last=True)
    ALGORITHM_REGISTRY.register(ppb.name, PPAC_BPRMF)
    df_merged = pd.read_csv('merged.csv')

    f1 = MinUsersPerItem(
    min_users_per_item=5,
    user_ix="user_id",
    item_ix="item_id"
    )

    df_merged = f1.apply(df_merged)

    f2 = MinItemsPerUser(
        min_items_per_user=10,
        user_ix="user_id",
        item_ix="item_id"
    )

    df_merged = f2.apply(df_merged)


    X = InteractionMatrix(df_merged, item_ix="item_id", user_ix="user_id")
    im = DummyDataset().load()
    scenario = WeakGeneralization(validation=True)
    scenario.split(im)

    pb = PipelineBuilder()
    
    # Use the configured search space when optimising the ItemKNN parameters
    pb.add_algorithm(ppb.name, 
                     params={
                             "num_components":100, 
                             "learning_rate":0.01, 
                             "lambda_h":0.001, 
                             "lambda_w":0.001, 
                             "validation_sample_size":100, 
                             "predict_topK":100, 
                             "save_best_to_file":True, 
                             "keep_last":True,
                             "batch_size": 128,
                             }
                             )
    pb.set_data_from_scenario(scenario)
    # Set NDCG@10 as the optimisation metric.
    # Since it is a metric to be maximized, the loss will be the negative of the NDCG, such that hyperopt can minimize it.
    pb.set_optimisation_metric('NDCGK', 10)
    pb.add_metric('NDCGK', 10)

    pipe = pb.build()
    pipe.run()

    df =pd.DataFrame.from_dict(pipe.get_metrics())
    print(df.head())