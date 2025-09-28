import math
import random
import logging
import numpy as np

import datasets
from dataclasses import dataclass
from typing import List, Union, Dict

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from arguments import DataArguments

logger = logging.getLogger(__name__)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: Union[datasets.Dataset, List[datasets.Dataset]],
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 2048,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.max_char_len = max_seq_len * 10

        assert isinstance(dataset, list), "Expected `dataset` to be a list."
        assert len(dataset) == 1 or len(dataset) == 2, "Expected `dataset` to be a list of length 1 or 2."

        if len(dataset) == 1:
            single_ds = dataset[0]
            assert len(single_ds) > 0, "Dataset is empty."

            sample_task_type = single_ds[0]["task_type"]
            assert sample_task_type in ("ir", "sts"), f"Unknown task_type: {sample_task_type}"

            if sample_task_type == "ir":
                self.ir_ds = single_ds
                self.len_ir = len(single_ds)

                self.sts_ds = None
                self.len_sts = 0
            elif sample_task_type == "sts":
                self.ir_ds = None
                self.len_ir = 0

                self.sts_ds = single_ds
                self.len_sts = len(single_ds)

        elif len(dataset) == 2:
            self.ir_ds = dataset[0]
            self.sts_ds = dataset[1]
            self.len_ir = len(self.ir_ds)
            self.len_sts = len(self.sts_ds)

        self.total_len = self.len_ir + self.len_sts

    def __len__(self):
        return self.total_len

    def get_positives(self, data_item):
        positives = data_item.get("positives", [])
        positive_scores = data_item.get("positive_scores", [-1.0] * len(positives))

        if len(positives) < self.args.positive_group_size:
            num = math.ceil(self.args.positive_group_size / len(positives))
            combined = list(zip(positives, positive_scores)) * num
            selected = random.sample(combined, self.args.positive_group_size)
            poss, scores = zip(*selected)
            poss, scores = list(poss), list(scores)
        else:
            indices = list(range(len(positives)))
            random.shuffle(indices)
            indices = indices[:self.args.positive_group_size]
            poss = [positives[i] for i in indices]
            scores = [positive_scores[i] for i in indices]
        return poss, scores

    def get_negatives(self, data_item):
        negatives = data_item.get("negatives", [])
        negative_scores = data_item.get("negative_scores", [-1.0] * len(negatives))

        if len(negatives) < self.args.negative_group_size:
            num = math.ceil(self.args.negative_group_size / len(negatives))
            combined = list(zip(negatives, negative_scores)) * num
            selected = random.sample(combined, self.args.negative_group_size)
            negs, scores = zip(*selected)
            negs, scores = list(negs), list(scores)
        else:
            indices = list(range(len(negatives)))
            random.shuffle(indices)
            indices = indices[:self.args.negative_group_size]
            negs = [negatives[i] for i in indices]
            scores = [negative_scores[i] for i in indices]
        return negs, scores

    def __getitem__(self, item: int):
        """
        Problems:
        If training for >1 epoch in unified mode, the same generative & embedding samples will
        always be in the same batch as the same index is used for both datasets.
        Solution:
        Don't train for >1 epoch by duplicating the dataset you want to repeat in the folder.
        Upon loading, each dataset is shuffled so indices will be different.
        """
        query, passages = None, None

        if item < self.len_ir:
            assert self.ir_ds is not None, f"Trying to access IR dataset at index {item}, but IR dataset is None"
            data_item = self.ir_ds[item]
        else:
            assert self.sts_ds is not None, f"Trying to access STS dataset at index {item - self.len_ir}, but STS dataset is None"
            data_item = self.sts_ds[item - self.len_ir]

        assert "query" in data_item and "positives" in data_item, f"该数据项缺少必要字段 query 或 positives, 索引 = {item}"

        query = data_item["query"]
        assert len(query) == 3, f"query 的长度不符合预期: {len(query)}, 索引 = {item}"
        query = [query[0], query[1], query[2][:self.max_char_len]]

        passages = []
        poss, scores = self.get_positives(data_item)

        for (pos, score) in zip(poss, scores):
            assert len(pos) == 3, f"正样本的长度不符合预期: {len(pos)}, 索引 = {item}"
            pos = [pos[0], pos[1], pos[2][:self.max_char_len]]
            passages.append((pos, score))

        negs, scores = self.get_negatives(data_item)

        for (neg, score) in zip(negs, scores):
            assert len(neg) == 3, f"负样本的长度不符合预期: {len(neg)}, 索引 = {item}"
            neg = [neg[0], neg[1], neg[2][:self.max_char_len]]
            passages.append((neg, score))

        task_type = data_item["task_type"]
        assert task_type in ["sts", "ir"], f"Invalid task type: {task_type}."
        return query, passages, task_type


@dataclass
class CustomCollator(DataCollatorWithPadding):
    query_max_len: int = 1024
    passage_max_len: int = 1024
    model_name_or_path: str = ""

    def build_query(self, text_list):
        """
        组装 instruction 和 text，并同时返回完整文本和指令部分的长度。
        返回: (full_text: str, instruction_len: int)
        """

        model_name_lower = self.model_name_or_path.lower()
        is_minicpm_or_e5 = "minicpm" in model_name_lower or "e5" in model_name_lower
        is_bge = "bge" in model_name_lower
        is_youtu = "youtu_" in model_name_lower

        assert model_name_lower, "model_name_or_path must be provided and cannot be an empty string."

        assert sum([is_minicpm_or_e5, is_bge, is_youtu]) == 1, \
            f"Model type ambiguity or unsupported. Exactly one of ('minicpm'/'e5', 'bge', 'youtu') " \
            f"must be specified in '{self.model_name_or_path}'."

        task_instruction, instance_instruction, text = [s.strip("\t\n :") for s in text_list]

        content_text = text
        task_text = ""

        if task_instruction:
            task_text = f"Instruction: {task_instruction} Query: "

        if is_minicpm_or_e5:
            task_text = f"<s>{task_text}"
            content_text += "</s>"
        elif is_bge:
            task_text = f"[CLS]{task_text}"
            content_text += "[SEP]"
        instruction_len = len(task_text)
        instance_text = instance_instruction[:512]

        full_text = f"{task_text}{instance_text}{content_text}"
        return full_text, instruction_len

    def build_passage(self, text_list):
        """
        组装 instruction 和 text，并同时返回完整文本和指令部分的长度。
        返回: (full_text: str, instruction_len: int)
        """

        model_name_lower = self.model_name_or_path.lower()
        is_minicpm_or_e5 = "minicpm" in model_name_lower or "e5" in model_name_lower
        is_bge = "bge" in model_name_lower
        is_youtu = "youtu_" in model_name_lower

        assert model_name_lower, "model_name_or_path must be provided and cannot be an empty string."

        assert sum([is_minicpm_or_e5, is_bge, is_youtu]) == 1, \
            f"Model type ambiguity or unsupported. Exactly one of ('minicpm'/'e5', 'bge', 'youtu') " \
            f"must be specified in '{self.model_name_or_path}'."

        __, instance_instruction, text = [s.strip("\t\n :") for s in text_list]

        content_text = text
        task_text = ""
        if is_minicpm_or_e5:
            task_text = "<s>"
            content_text += "</s>"
        elif is_bge:
            task_text = "[CLS]Passage: "
            content_text += "[SEP]"

        instruction_len = len(task_text)
        instance_text = instance_instruction[:512]

        full_text = f"{task_text}{instance_text}{content_text}"
        return full_text, instruction_len

    def create_text_mask(self, encodings: Dict[str, torch.Tensor], instruction_char_lens: List[int]) -> torch.Tensor:
        """
        根据 offset_mapping 和指令的字符长度，创建精确的 text_mask。
        1 表示 text token, 0 表示 instruction/padding token。
        """
        offsets = encodings.pop("offset_mapping")
        device = offsets.device

        instruction_lens_tensor = torch.tensor(
            instruction_char_lens,
            device=device
        ).unsqueeze(1)

        end_offsets = offsets[:, :, 1]

        text_mask = (end_offsets > instruction_lens_tensor).to(encodings["attention_mask"].dtype)

        return text_mask * encodings["attention_mask"]

    def process_passages(self, passage_groups, task_types):
        d_results, scores = [], []
        for group_item in passage_groups:
            for passage_item in group_item:
                assert isinstance(passage_item, tuple) and len(passage_item) == 2, \
                    f"Unprocessable passage format: {type(passage_item)}"

                scores.append(passage_item[1])

                if task_types[0] == "ir":
                    doc = self.build_passage(passage_item[0])
                else:
                    doc = self.build_query(passage_item[0])
                d_results.append(doc)

        passage_full_text, d_instruction_char_lens = list(zip(*d_results))
        return passage_full_text, d_instruction_char_lens, scores

    def __call__(self, features):
        queries = [f[0] for f in features]
        passage_groups = [f[1] for f in features]
        task_types = [f[2] for f in features]

        assert len(set(task_types)) == 1, f"Multiple task_types appeared in this batch: {set(task_types)}"

        q_results = [self.build_query(item) for item in queries]
        query_full_text, q_instruction_char_lens = list(zip(*q_results))

        passage_full_text, d_instruction_char_lens, scores = self.process_passages(passage_groups, task_types)

        q_encodings = self.tokenizer(
            query_full_text,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        d_encodings = self.tokenizer(
            passage_full_text,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        q_text_mask = self.create_text_mask(q_encodings, q_instruction_char_lens)
        d_text_mask = self.create_text_mask(d_encodings, d_instruction_char_lens)

        processed_features = {
            "query": q_encodings,
            "passage": d_encodings,
        }

        processed_features["query"]["text_mask"] = q_text_mask
        processed_features["passage"]["text_mask"] = d_text_mask
        processed_features["scores"] = torch.tensor(scores, dtype=torch.float)
        processed_features["task_type"] = task_types

        return processed_features
