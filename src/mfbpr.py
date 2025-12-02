import numpy as np
import torch
import torch.nn as nn
# from recpack.algorithms.base import Algorithm
from recpack.algorithms.bprmf import BPRMF
from recpack.algorithms.loss_functions import bpr_loss
from recpack.algorithms.samplers import BootstrapSampler

class PPAC_BPRMF(BPRMF):
    """
    Personalized Pairwise Adaptive Collaborative Bayesian Personalized Ranking Matrix Factorization (PPAC-BPRMF)
    
    This model extends the traditional BPR-MF by incorporating personalized pairwise adaptive learning rates.
    It adjusts the learning rate for each user-item interaction based on historical feedback, allowing for
    more personalized recommendations.
    
    Parameters
    ----------
    num_users : int
        Number of users in the dataset.
    num_items : int
        Number of items in the dataset.
    embedding_dim : int
        Dimensionality of the user and item embeddings.
    lr : float
        Base learning rate for optimization.
    reg : float
        Regularization term for embeddings.
    """
    print("Initializing PPAC_BPRMF")
    def _init_model(self, X):
        super()._init_model(X)

        print(self.batch_size)
        
        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size
        )

        # compute popularity of items from user-item interaction matrix X
        popularity = X.sum(axis=0).A1 

        popularity = popularity + 1e-6  # avoid division by zero
        self._popularity = popularity

        # PPAC weight: inverse of popularity
        # w(i) = 1 / log(pop(i) + 1)

        ppac_weights = 1.0 / np.log(self._popularity + 1)

        # normalize weights
        ppac_weights = ppac_weights / np.max(ppac_weights)

        self._ppac_weights = torch.tensor(ppac_weights, dtype=torch.float32, device=self.device)
    
    def _compute_loss(self, positive_sim, negative_sim, pos_items, neg_items):
        """
        Compute the PPAC-BPR loss with personalized pairwise adaptive weights.
        
        Parameters
        ----------
        positive_sim : torch.Tensor
            Similarity scores for positive user-item pairs.
        negative_sim : torch.Tensor
            Similarity scores for negative user-item pairs.
        pos_items : torch.Tensor
            Indices of positive items.
        neg_items : torch.Tensor
            Indices of negative items.
        
        Returns
        -------
        torch.Tensor
            Computed PPAC-BPR loss.
        """
        bpr_loss_value = bpr_loss(positive_sim, negative_sim)

        
        pos_w = torch.tensor(
            self._ppac_weights[pos_items.cpu()], device=self.device
        )
        neg_w = torch.tensor(
            self._ppac_weights[neg_items.cpu()], device=self.device
        )

        # Symmetric weighting
        weight = (pos_w + neg_w) / 2.0

        # Apply PPAC weighting
        loss = weight * bpr_loss_value

        # Add regularization
        loss += (
            self.lambda_h * self.model_.item_embedding_.weight.norm()
            + self.lambda_w * self.model_.user_embedding_.weight.norm()
        )

        return loss.mean()


    def _train_epoch(self, X):
        """
        Overridden to pass pos/neg item IDs to PPAC-weighted loss.
        """

        losses = []
        self.model_.train()

        for users, target_items, neg_items in self.sampler.sample(X, sample_size=self.sample_size):
            users = users.to(self.device)
            target_items = target_items.to(self.device)
            neg_items = neg_items.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()

            # Forward pass using the model
            pos_sim = self.model_.forward(users, target_items).diag()
            neg_sim = self.model_.forward(users, neg_items).diag()

            # PPAC-weighted loss
            loss = self._compute_loss(pos_sim, neg_sim, target_items, neg_items)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

