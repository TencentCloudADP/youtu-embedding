import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from scipy.stats import rankdata
import torch.distributed as dist
import torch.distributed.nn as dist_nn


class ContrastiveLoss:
    def __init__(
        self,
        task_type: str = "sts",
        threshold: float = 1.0,
        temperature: float = 1.0,
        positive_group_size: int = 1,
        negative_group_size: int = 1,
        negatives_cross_device: bool = False,
    ) -> None:
        self.task_type = task_type
        self.threshold = threshold
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        self.ir_mode = "all"
        self.positive_group_size = positive_group_size
        self.negative_group_size = negative_group_size
        assert self.ir_mode in ["sample", "all"], f"Invalid ir_mode: {self.ir_mode}"
        assert self.task_type in ["sts", "ir"], f"Invalid task_type: {self.task_type}"

        if self.negatives_cross_device and not dist.is_initialized():
            raise ValueError("negatives_cross_device=True requires distributed training")

    def __call__(
        self,
        q_reps: torch.Tensor,
        p_reps: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.negatives_cross_device:
            q_reps = self._gather_tensor(q_reps)
            p_reps = self._gather_tensor(p_reps)
            if y is not None:
                y = self._gather_tensor(y)

        if self.task_type.lower() == "sts":
            return self._loss_sts(q_reps, p_reps, y)
        elif self.task_type.lower() == "ir":
            return self._loss_ir(q_reps, p_reps)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _loss_sts(
        self,
        q_reps: torch.Tensor,
        p_reps: torch.Tensor,
        y: Optional[torch.Tensor]
    ) -> torch.Tensor:
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        group_size = p_reps.size(0) // q_reps.size(0)

        if y is None:
            valid_mask = torch.ones_like(q_reps[:, 0], dtype=torch.bool)
        else:
            positive_scores = y[::group_size]
            valid_mask = positive_scores >= self.threshold

        num_valid = valid_mask.sum()

        if num_valid == 0:
            return torch.tensor(0.0, device=q_reps.device, requires_grad=True)

        targets = (torch.arange(q_reps.size(0), device=q_reps.device) * group_size)[valid_mask]
        scores = scores[valid_mask]

        return self.cross_entropy(scores, targets)

    def _loss_ir(
        self, 
        q_reps: torch.Tensor,
        p_reps: torch.Tensor
    ) -> torch.Tensor:
        batch_size = q_reps.shape[0]
        group_size = self.positive_group_size + self.negative_group_size
        base = torch.arange(batch_size, device=q_reps.device) * group_size  # (batch_size,)

        if self.ir_mode == "sample":
            local_positive_index = torch.randint(self.positive_group_size, (batch_size,), device=q_reps.device)  # (batch_size,)
            global_positive_index = base + local_positive_index  # (batch_size,)

            negative_offsets = torch.arange(self.positive_group_size, group_size, device=q_reps.device)  # (nagative_group_size,)
            global_negative_index = base.unsqueeze(1) + negative_offsets  # (batch_size, negative_group_size)

            # (batch_size, 1 + negative_group_size) => (batch_size * (1 + negative_group_size),)
            gathered_index = torch.cat([global_positive_index.unsqueeze(1), global_negative_index], dim=1).view(-1)
            filtered_p_reps = p_reps[gathered_index]

            new_group_size = 1 + self.negative_group_size
            scores = self.compute_similarity(q_reps, filtered_p_reps) / self.temperature
            targets = (torch.arange(q_reps.size(0), device=q_reps.device) * new_group_size)
            return self.cross_entropy(scores, targets)
        elif self.ir_mode == "all":
            negative_offsets = torch.arange(self.positive_group_size, group_size, device=q_reps.device)
            global_negative_index = base.unsqueeze(1) + negative_offsets  # (batch_size, negative_group_size)

            total_loss = 0.0
            new_group_size = 1 + self.negative_group_size
            for positive_offset in range(self.positive_group_size):
                global_positive_index = base + positive_offset
                
                gathered_index = torch.cat([global_positive_index.unsqueeze(1), global_negative_index], dim=1).view(-1)
                filtered_p_reps = p_reps[gathered_index]

                scores = self.compute_similarity(q_reps, filtered_p_reps) / self.temperature
                targets = (torch.arange(q_reps.size(0), device=q_reps.device) * new_group_size)
                loss = self.cross_entropy(scores, targets)
                total_loss += loss
            return total_loss / self.positive_group_size
        else:
            raise ValueError(f"Invalid ir_mode: {self.ir_mode}")

    @staticmethod
    def compute_similarity(q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    @staticmethod
    def _gather_tensor(t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return None
        return torch.cat(dist_nn.all_gather(t), dim=0)


class PearsonCorrelationLoss(nn.Module):
    def __init__(self, negatives_cross_device: bool = False):
        super().__init__()
        self.negatives_cross_device = negatives_cross_device
        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.world_size = dist.get_world_size()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.negatives_cross_device:
            x = self._dist_gather_tensor(x)
            y = self._dist_gather_tensor(y)

        if y.numel() == 0: 
            return torch.tensor(0.0, device=y.device, requires_grad=True)
        elif y.min() == y.max(): 
            return torch.tensor(0.0, device=y.device, requires_grad=True)

        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        var_x = torch.var(x, unbiased=False)
        var_y = torch.var(y, unbiased=False)
        pearson = torch.mean((x - mean_x) * (y - mean_y)) / (torch.sqrt(var_x) * torch.sqrt(var_y))
        return (-pearson + 1)

    def _dist_gather_tensor(self, t: torch.Tensor):
        if t is None:
            return None
        return torch.cat(dist_nn.all_gather(t), dim=0)


class RankKLDivergenceLoss(nn.Module):
    """
    Calculates KL divergence loss based on fair ranking (handles ties correctly),
    aligning better with Spearman's rank correlation.

    Tied similarity scores are assigned their average rank to ensure they are
    treated equally by the loss function.
    """
    def __init__(
        self, 
        temperature: float = 0.05,
        negatives_cross_device: bool = False,
        reduction: str = "batchmean"
    ):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        
        if self.negatives_cross_device and not dist.is_initialized():
            raise ValueError("negatives_cross_device=True requires distributed training")

    def _get_average_ranks(self, x: torch.Tensor) -> torch.Tensor:
        ranks_1_based = rankdata(-x.cpu().float().numpy(), method='average')
        return torch.from_numpy(ranks_1_based - 1).to(x.device) 

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        if self.negatives_cross_device:
            predict_similarity = self._dist_gather_tensor(predict_similarity)
            true_similarity = self._dist_gather_tensor(true_similarity)
        
        if predict_similarity.numel() == 0 or true_similarity.numel() == 0:
            return torch.tensor(0.0, device=predict_similarity.device, requires_grad=True)
        
        ranks = self._get_average_ranks(true_similarity)
        n = true_similarity.numel()

        if n > 1: 
            ideal_scores_normalized = (n - 1 - ranks).float() / (n - 1)
        else:
            ideal_scores_normalized = torch.tensor([0.0], device=ranks.device)

        true_probs = F.softmax(ideal_scores_normalized / self.temperature, dim=0)
        log_pred_probs = F.log_softmax(predict_similarity / self.temperature, dim=0)

        kl_loss = F.kl_div(log_pred_probs, true_probs, reduction=self.reduction)
        return kl_loss

    def _dist_gather_tensor(self, t: torch.Tensor):
        if t is None:
            return None
        return torch.cat(dist_nn.all_gather(t), dim=0)


class PROLoss(nn.Module):
    """
    An implementation of the PRO loss from "Large Language Model based 
    Long-tail Query Rewriting in Taobao Search", adapted for Semantic
    Textual Similarity (STS) tasks.

    This list-wise loss function encourages the model's predicted similarity
    scores to follow the same ranking as the ground-truth scores. It does so
    by decomposing the ranking problem into N-1 sub-problems, where for each
    step k, the k-th item is treated as the positive example and all subsequent
    items are negatives. The loss is weighted by the difference in ground-truth
    scores, placing more emphasis on correctly ranking pairs with larger
    disparities.
    """
    def __init__(self, eps: float = 1e-8, temperature: float = 0.05, negatives_cross_device: bool = False):
        """
        Args:
            eps (float): A small epsilon value to ensure numerical stability and
                         handle comparisons, e.g., to filter for strictly
                         positive score differences.
        """
        super().__init__()

        self.eps = eps
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        
        if self.negatives_cross_device and not dist.is_initialized():
            raise ValueError("negatives_cross_device=True requires distributed training")

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        """
        Calculates the PRO loss.

        Args:
            predict_similarity (torch.Tensor): A 1D tensor of predicted similarity
                scores from the model. Shape: (N,).
            true_similarity (torch.Tensor): A 1D tensor of ground-truth similarity
                scores. Shape: (N,).

        Returns:
            torch.Tensor: The calculated PRO loss, a scalar tensor.
        """
        if self.negatives_cross_device:
            predict_similarity = self._dist_gather_tensor(predict_similarity)
            true_similarity = self._dist_gather_tensor(true_similarity)
        
        if predict_similarity.numel() == 0 or true_similarity.numel() < 2:
            return torch.tensor(0.0, device=predict_similarity.device, requires_grad=True)

        predict_similarity = predict_similarity / self.temperature
        sorted_indices = torch.argsort(true_similarity, descending=True)

        y_true_sorted = true_similarity[sorted_indices]
        y_pred_sorted = predict_similarity[sorted_indices]

        loss_count = 0
        total_log_probs = 0.0
        batch_size = y_true_sorted.size(0)

        for k in range(batch_size - 1):
            y_k_true = y_true_sorted[k]
            y_k_pred = y_pred_sorted[k]
            y_tail_true = y_true_sorted[k + 1:]
            y_tail_pred = y_pred_sorted[k + 1:]
            mask = (y_k_true - y_tail_true) > self.eps
            
            if not mask.any():
                continue

            y_neg_true = y_tail_true[mask]
            y_neg_pred = y_tail_pred[mask]
            weights_neg = y_k_true - y_neg_true
            
            weight_k = weights_neg.max()
            logit_k = y_k_pred * weight_k
            logits_neg = y_neg_pred * weights_neg
            all_logits = torch.cat([logit_k.unsqueeze(0), logits_neg])
            
            log_prob_k = F.log_softmax(all_logits, dim=0)[0]
            total_log_probs += log_prob_k
            loss_count += 1

        if loss_count == 0:
            return torch.tensor(0.0, device=predict_similarity.device, requires_grad=True)

        pro_loss = -total_log_probs / loss_count
        
        return pro_loss

    def _dist_gather_tensor(self, t: torch.Tensor):
        if t is None:
            return None
        return torch.cat(dist_nn.all_gather(t), dim=0)
