import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
from torch import Tensor
from transformers.file_utils import ModelOutput
from transformers import AutoModel, AutoModelForCausalLM

from loss import ContrastiveLoss
from loss import PearsonCorrelationLoss, RankKLDivergenceLoss, PROLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class TrainModel(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        pooling_method: str = 'mean',
        normalized: bool = True,
        projection: int = None,
        attn: str = 'bbcc',
        temperature: float = 1.0,
        ir_negatives_cross_device: bool = False,
        sts_negatives_cross_device: bool = False,
        multi_layer_loss: bool = False,
        positive_group_size: int = 1,
        negative_group_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        if "youtu_" in model_name_or_path:
            base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, output_hidden_states=True, **kwargs)
            self.model = base_model.model
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, output_hidden_states=True, **kwargs)
        print(f"Created TrainLM: {self.model.dtype} dtype, {pooling_method} pool, {attn} attn")

        self.projection = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=int(projection),
            dtype=self.model.dtype
        ) if projection is not None else None

        self.attn = attn
        self.normalized = normalized
        self.pooling_method = pooling_method
        self.multi_layer_loss = multi_layer_loss
        self.positive_group_size = positive_group_size
        self.negative_group_size = negative_group_size
        self.config = self.model.config # Required for accelerate DeepSpeed integration

        # STS Loss Functions
        self.pro_loss_fun = PROLoss(temperature=0.05, negatives_cross_device=sts_negatives_cross_device)
        self.pearson_loss_fn = PearsonCorrelationLoss(negatives_cross_device=sts_negatives_cross_device)
        self.kl_div_loss_fn = RankKLDivergenceLoss(temperature=0.05, negatives_cross_device=sts_negatives_cross_device)
        self.sts_infonce_loss_fn = ContrastiveLoss(task_type="sts", threshold=1.0, temperature=temperature, negatives_cross_device=sts_negatives_cross_device)

        # IR Loss Functions
        self.ir_infonce_loss_fn = ContrastiveLoss(task_type="ir", temperature=temperature, positive_group_size=positive_group_size,
                                                  negative_group_size=negative_group_size, negatives_cross_device=ir_negatives_cross_device)


    def encode_with_multi_layer_loss(self, features):
        # Clone to avoid modifying the original tensor
        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}
        text_mask = features['text_mask'] if 'text_mask' in features else None

        if self.attn[:2] == "bb" and "youtu_" in self.model_name_or_path:
            kwargs['is_causal'] = False

        outputs = self.model(**kwargs)
        n = len(outputs.hidden_states)
        last_hidden_state = outputs.last_hidden_state
        certain_hidden_state = outputs.hidden_states[n // 2]

        if self.projection is not None:
            certain_hidden_state = self.projection(certain_hidden_state)
            last_hidden_state = self.projection(last_hidden_state)

        if text_mask is not None and self.pooling_method in ['mean', 'weightedmean']:
            certain_reps = self.pooling(certain_hidden_state, text_mask)
            last_reps = self.pooling(last_hidden_state, text_mask)
        else:
            certain_reps = self.pooling(certain_hidden_state, attention_mask)
            last_reps = self.pooling(last_hidden_state, attention_mask)

        if self.normalized:
            in_dtype = last_reps.dtype
            certain_reps = torch.nn.functional.normalize(certain_reps, dim=-1).contiguous().to(in_dtype)
            last_reps = torch.nn.functional.normalize(last_reps, dim=-1).contiguous().to(in_dtype)

        return certain_reps.contiguous(), last_reps.contiguous()

    def encode_wo_multi_layer_loss(self, features):
        # Clone to avoid modifying the original tensor
        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}
        text_mask = features['text_mask'] if 'text_mask' in features else None

        if self.attn[:2] == "bb" and "youtu_" in self.model_name_or_path:
            kwargs['is_causal'] = False

        last_hidden_state = self.model(**kwargs).last_hidden_state

        if self.projection is not None:
            last_hidden_state = self.projection(last_hidden_state)

        if text_mask is not None and self.pooling_method in ['mean', 'weightedmean']:
            reps = self.pooling(last_hidden_state, text_mask)
        else:
            reps = self.pooling(last_hidden_state, attention_mask)

        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized:
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)

        return reps.contiguous()

    def encode(self, features):
        if not self.multi_layer_loss:
            return self.encode_wo_multi_layer_loss(features)
        else:
            return self.encode_with_multi_layer_loss(features)

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        scores: torch.Tensor = None,
        task_type: List = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ):
        if len(set(task_type)) > 1:
            raise ValueError(f"Multiple task_types appeared in this batch: {set(task_type)}")

        task_type = task_type[0]
        batch_size = query['input_ids'].shape[0]
        group_size = self.positive_group_size + self.negative_group_size

        if task_type == "sts":
            indices_to_keep = torch.arange(batch_size, device=passage['input_ids'].device) * group_size

            scores = scores[indices_to_keep]
            passage['text_mask'] = passage['text_mask'][indices_to_keep]
            passage['input_ids'] = passage['input_ids'][indices_to_keep]
            passage['attention_mask'] = passage['attention_mask'][indices_to_keep]
            assert -1 not in scores, f"STS scores should not contain -1, got {scores}"

        if q_grad:
            if self.multi_layer_loss:
                certain_q_reps, q_reps = self.encode(query)
            else:
                q_reps = self.encode(query)
        else:
            with torch.no_grad():
                q_reps = self.encode(query)

        if p_grad:
            if self.multi_layer_loss:
                certain_p_reps, p_reps = self.encode(passage)
            else:
                p_reps = self.encode(passage)
        else:
            with torch.no_grad():
                p_reps = self.encode(passage)

        if task_type == "sts":
            if self.multi_layer_loss:
                loss_infonce = self.sts_infonce_loss_fn(certain_q_reps, certain_p_reps, scores)

                if torch.cuda.current_device() == 0:
                    print(f"STS Infonce Loss: {loss_infonce.item():.4f}")
            else:
                loss_infonce = 0.0

            if not self.normalized:
                print('need to normalize before computing cosine similarity')
                q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)

            sim = torch.sum(q_reps * p_reps, dim=-1)
            loss_pearson = 2 * self.pearson_loss_fn(sim, scores)
            loss_pro = 0.5 * self.pro_loss_fun(sim, scores)
            loss_kl = 5 * self.kl_div_loss_fn(sim, scores)

            loss = loss_infonce + loss_pro + loss_pearson + loss_kl

            if torch.cuda.current_device() == 0:
                print(f"STS Pearson Loss: {loss_pearson.item():.4f}")
                print(f"STS PRO Loss: {loss_pro.item():.4f}")
                print(f"STS KL Loss: {loss_kl.item():.4f}")
                print(f"STS Embedding Loss: {loss.item():.4f}")

            loss = loss.to(torch.float32)
        elif task_type == "ir":
            if not self.normalized:
                q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)

            loss_infonce = self.ir_infonce_loss_fn(q_reps, p_reps)
            loss = loss_infonce

            if torch.cuda.current_device() == 0:
                print(f"IR InfoNCE Loss: {loss_infonce.item():.4f}")
                print(f"IR Total Loss: {loss.item():.4f}")

            loss = loss.to(torch.float32)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Also return q_reps in case of GradCache
        return TrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
        )

    def pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False) -> torch.Tensor:
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)

        if self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        elif self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        else:
            raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")

        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
