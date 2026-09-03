import numpy as np
import torch
import torch.nn.functional as F
from recpack.algorithms.bprmf import BPRMF, MFModule
from recpack.algorithms.loss_functions import bpr_loss_wrapper
from recpack.algorithms.samplers import BootstrapSampler
from recpack.algorithms.util import sample_rows
from recpack.matrix import to_csr_matrix
from scipy.sparse import lil_matrix
from torch import nn, optim

from src.metrics import cal_global_nov, cal_local_nov


def _l2_loss_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean, over this batch's rows only, of each row's squared norm / 2.
    Matches the PPAC paper's reference implementation's L2 term"""
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1) / 2.0)


def _log_minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """Compress a heavy-tailed count vector with log1p, then rescale to [0, 1].

    Plain L2- or min-max-normalizing raw popularity counts leaves the long
    tail of items clustered near 0 (a handful of blockbusters dominate the
    scale), which saturates a Sigmoid trained to regress toward it and kills
    its gradient. log1p compresses the tail before rescaling, so the target
    distribution actually spans the Sigmoid's responsive range.
    """
    log_x = torch.log1p(x)
    return (log_x - log_x.min()) / (log_x.max() - log_x.min())


def _log_minmax_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Row-wise version of `_log_minmax_normalize`, for a (users x items)
    matrix: each user's row is independently log1p-compressed and rescaled
    to [0, 1], so a user's single most locally-popular item reaches 1.0 -
    the same achievable ceiling as `global_pop`'s single most globally-popular
    item. A plain global min-max (or the row-L2-normalize this replaced)
    leaves most users' rows far below that ceiling, so `beta * real_global`
    (which does reach it for any popular item) can never be outweighed by
    `gamma * real_local` at any shared gamma/beta magnitude - see PPAC_BPRMF's
    _batch_predict.

    Rows with no signal (a user with zero Jaccard-overlapping neighbors)
    would divide 0/0; they're left at all-zero instead of NaN.
    """
    log_x = torch.log1p(x)
    row_min = log_x.min(dim=1, keepdim=True).values
    row_max = log_x.max(dim=1, keepdim=True).values
    row_range = row_max - row_min
    has_range = row_range > 0
    safe_range = torch.where(has_range, row_range, torch.ones_like(row_range))
    normalized = (log_x - row_min) / safe_range
    return torch.where(has_range, normalized, torch.zeros_like(normalized))


def _best_available_device() -> torch.device:
    """this function is to maintain compatibility with Apple and Google Collab"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PPAC_BPRMF(BPRMF):
    """
    Personalized Popularity-Aware Collaborative BPR-MF (PPAC-BPRMF)

    Extends recpack's own `BPRMF` with popularity-aware predictions using both
    local (user-specific) and global novelty patterns, following the PPAC
    framework. The model learns to predict item novelty/popularity at both
    global and local levels, and uses these predictions to adjust
    recommendation scores.



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
    """

    def __init__(
        self,
        num_components=128,
        batch_size=512,
        max_epochs=8,
        learning_rate=0.005,
        gamma=0.1,
        beta=0.1,
        reg_coe=1e-3,
        l2_coe=1e-4,
        stopping_criterion="bpr",
        save_best_to_file=False,
        keep_last=False,
        predict_topK=100,
        validation_sample_size=200,
        seed=None,
        optimizer_cls: type = optim.Adam,
        lr_end_factor: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            num_components=num_components,
            batch_size=batch_size,
            lambda_h=0.0,
            lambda_w=0.0,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            stopping_criterion=stopping_criterion,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
            **kwargs,
        )

        self.device = _best_available_device()

        self.gamma = gamma
        self.beta = beta
        self.reg_coe = reg_coe
        self.l2_coe = l2_coe

        self.optimizer_cls = optimizer_cls
        self.lr_end_factor = lr_end_factor

        # novelty-aware components from PPAC Framework
        self.local_pred = None
        self.global_pred = None
        self.global_pop = None
        self.local_pop = None
        self.reg_loss_fn = nn.MSELoss()
        self.f = nn.Sigmoid()

        self.losses_ = None

        # Per-epoch history for plotting
        self.train_losses_ = []
        self.val_losses_ = []
        self.val_bpr_losses_ = []
        self.val_cf_losses_ = []

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
        """Initialize model, optimizer/scheduler and novelty-aware components.

        Builds the base MF model and optimizer (Adam as default) with a linear LR schedule.
        """
        num_users, num_items = X.shape
        self.model_ = MFModule(num_users, num_items, num_components=self.num_components).to(self.device)

        self.optimizer = self.optimizer_cls(self.model_.parameters(), lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=self.lr_end_factor, total_iters=self.max_epochs
        )

        # Initialize sampler
        self.sampler = BootstrapSampler(num_negatives=1, batch_size=self.batch_size)

        # Convert sparse matrix to train_records format
        _, num_items = X.shape
        train_records = self._convert_sparse_to_train_records(X)

        _, global_pop_counts = cal_global_nov(train_records, num_items)
        self.global_pop = _log_minmax_normalize(global_pop_counts.float()).to(self.device)

        _, local_pop_counts = cal_local_nov(X)
        self.local_pop = _log_minmax_normalize_rows(local_pop_counts.float()).to(self.device)

        # Initialize novelty prediction networks
        latent_dim = self.num_components

        # Local novelty predictor (user-item interaction patterns)
        self.local_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.LeakyReLU(), nn.Linear(latent_dim // 2, latent_dim)
        ).to(self.device)

        # Global novelty predictor (overall item novelty)
        self.global_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0),
        ).to(self.device)

        # Add predictors to optimizer
        predictor_params = list(self.local_pred.parameters()) + list(self.global_pred.parameters())
        self.optimizer.add_param_group({"params": predictor_params})

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

        # Predict local novelty
        usr_ci_emb = F.normalize(self.local_pred(user_emb), p=2, dim=-1)
        pos_ci_emb = F.normalize(self.local_pred(pos_item_emb), p=2, dim=-1)
        neg_ci_emb = F.normalize(self.local_pred(neg_item_emb), p=2, dim=-1)

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
            self.reg_loss_fn(pos_ci_local, self.local_pop[users, pos_items])
            + self.reg_loss_fn(neg_ci_local, self.local_pop[users, neg_items])
        ) / 2

        global_reg_loss = (
            self.reg_loss_fn(pos_ci_global, self.global_pop[pos_items])
            + self.reg_loss_fn(neg_ci_global, self.global_pop[neg_items])
        ) / 2

        reg_loss = local_reg_loss + global_reg_loss

        # L2 regularization on embeddings
        l2_loss = _l2_loss_mean(user_emb) + _l2_loss_mean(pos_item_emb) + _l2_loss_mean(neg_item_emb)

        # Total loss (matching original PPAC formula)
        total_loss = cf_loss + self.reg_coe * reg_loss + self.l2_coe * l2_loss

        return total_loss

    def _cf_bpr_loss_on(self, X_true) -> float:
        """BPR loss on X_true's sampled positive/negative pairs, scored with
        the exact same novelty-modulated CF formula _compute_loss's cf_loss
        trains on above.
        """
        self.model_.eval()
        self.local_pred.eval()
        self.global_pred.eval()

        sampler = BootstrapSampler(num_negatives=1, batch_size=self.batch_size)
        losses = []
        with torch.no_grad():
            for users, pos_items, neg_items in sampler.sample(X_true):
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.squeeze(-1).to(self.device)

                user_emb = self.model_.user_embedding_(users)
                pos_item_emb = self.model_.item_embedding_(pos_items)
                neg_item_emb = self.model_.item_embedding_(neg_items)

                pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
                neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)

                usr_ci_emb = F.normalize(self.local_pred(user_emb), p=2, dim=-1)
                pos_ci_emb = F.normalize(self.local_pred(pos_item_emb), p=2, dim=-1)
                neg_ci_emb = F.normalize(self.local_pred(neg_item_emb), p=2, dim=-1)

                pos_ci_local = self.f(torch.mul(usr_ci_emb, pos_ci_emb).sum(1))
                neg_ci_local = self.f(torch.mul(usr_ci_emb, neg_ci_emb).sum(1))

                pos_ci_global = self.global_pred(pos_item_emb)
                neg_ci_global = self.global_pred(neg_item_emb)

                pos_scores = pos_scores * (pos_ci_local * pos_ci_global)
                neg_scores = neg_scores * (neg_ci_local * neg_ci_global)

                losses.append(F.softplus(neg_scores - pos_scores).mean().item())

        self.model_.train()
        self.local_pred.train()
        self.global_pred.train()

        return float(np.mean(losses))

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

        mean_loss = np.mean(losses)
        self.train_losses_.append(mean_loss)

        return mean_loss

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

            # Predict novelty patterns. L2-normalized to match
            # _compute_loss's fix - see its comment for why.
            user_ci_emb = F.normalize(self.local_pred(user_emb), p=2, dim=-1)
            item_ci_emb = F.normalize(self.local_pred(item_emb), p=2, dim=-1)

            pred_local = self.f(torch.matmul(user_ci_emb, item_ci_emb.T))
            pred_global = self.global_pred(item_emb).expand(scores.shape)

            # Get real novelty values
            real_local = self.local_pop[user_tensor]
            real_global = self.global_pop.expand(scores.shape)

            # Combine CF scores with novelty awareness (matching original PPAC formula)
            scores = scores * (pred_local * pred_global) + self.gamma * real_local + self.beta * real_global

            result = lil_matrix(X.shape)
            result[users] = scores.cpu().numpy()

            return result.tocsr()

    def _evaluate(self, val_in, val_out):
        """Same as TorchMLAlgorithm._evaluate, except it also records the raw
        per-epoch validation score for checking over/underfitting."""

        val_in = self._transform_predict_input(val_in)
        val_out = to_csr_matrix(val_out)

        if self.validation_sample_size:
            val_in, val_out = sample_rows(val_in, val_out, sample_size=self.validation_sample_size)

        X_pred_cpu = self._predict(val_in)

        current_value = self.stopping_criterion.loss_function(val_out, X_pred_cpu, **self.stopping_criterion.kwargs)
        self.val_losses_.append(current_value)

        self.val_bpr_losses_.append(bpr_loss_wrapper(val_out, X_pred_cpu))

        self.val_cf_losses_.append(self._cf_bpr_loss_on(val_out))

        better = self.stopping_criterion.update(val_out, X_pred_cpu)
        if better and not self.keep_last:
            self._save_best()
