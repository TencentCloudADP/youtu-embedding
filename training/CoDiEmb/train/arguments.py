import os
from typing import Optional
from dataclasses import dataclass, field

from transformers.training_args import *
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    pooling_method: str = field(
        default="mean", metadata={"help": "Pooling method for sentences"}
    )
    normalized: bool = field(
        default=True
    )
    attn: str = field(
        default="bbcc",
        metadata={
            "help": "bidirectional/causal attn for emb inst., emb sample, gen inst., gen sample"
            " e.g. bbcc is bidirectional over both emb inst. & sample but causal over gen inst. & sample"
            " cccc is causal over all; bccc is bidirectional over emb inst. but causal over rest etc."
        },
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "eager/sdpa/flash_attention_2"}
    )
    projection: int = field(
        default=None,
        metadata={"help": "Optional linear learned embedding down projection"},
    )
    multi_layer_loss: bool = field(
        default=False,
        metadata={"help": "倘若开启，则倒数第二层应用 infonce loss，最后一层应用 list-wise loss"}
    )


@dataclass
class DataArguments:
    ir_train_data: str = field(
        default=None,
        metadata={
            "help": "Path to folder or file with ir training data."
        },
    )
    sts_train_data: str = field(
        default=None,
        metadata={
            "help": "Path to folder or file with sts training data."
        },
    )
    positive_group_size: int = field(
        default=3,
        metadata={"help": "Number of positives for a query in training"}
    )
    negative_group_size: int = field(
        default=8,
        metadata={"help": "Number of negatives for a query in training"}
    )
    data_sampler: str = field(
        default="dynamic",
        metadata={
            "help": "Data sampler to use for training. Can be one of 'single', 'mixed', 'dynamic'."
        },
    )
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum tokens for the query. Sequences longer"
            " than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum tokens for passages (positives & negatives). Sequences longer"
            " than this will be truncated, sequences shorter will be padded."
        },
    )
    max_example_num_per_dataset: int = field(
        default=100_000_000,
        metadata={"help": "the max number of examples for each dataset"},
    )
    num_samples: Optional[str] = field(
        default=None,
        metadata={"help": "path to json with number of samples per dataset"},
    )
    query_instruction: str = field(
        default="Query: ", metadata={"help": "prepend instruction to query"}
    )
    passage_instruction: str = field(
        default="Passage: ", metadata={"help": "prepend instruction to document"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "path to cache directory"}
    )
    def __post_init__(self):
        if not self.ir_train_data and not self.sts_train_data:
            raise ValueError("At least one of ir_train_data or sts_train_data must be provided")

        if self.ir_train_data and not os.path.exists(self.ir_train_data):
            raise FileNotFoundError(f"IR train data path does not exist: {self.ir_train_data}")

        if self.sts_train_data and not os.path.exists(self.sts_train_data):
            raise FileNotFoundError(f"STS train data path does not exist: {self.sts_train_data}")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    skip_filter_too_long_instruction: bool = field(
        default=False, metadata={"help": "skip filter too long instructions"}
    )
    ir_negatives_cross_device: bool = field(
        default=False,
        metadata={
            "help": "Share the negatives across all GPUs. This argument will extend the number of negatives."
        },
    )
    sts_negatives_cross_device: bool = field(
        default=False,
        metadata={
            "help": "Share the negatives across all GPUs. This argument will extend the number of negatives."
        },
    )
    ir_per_device_batch_size: int = field(
        default=32,
        metadata={
            "help": "IR 任务中，每台 GPU 处理的 batch size。由于 IR 任务文本较长，通常使用较小的 batch size。"
        },
    )
    sts_per_device_batch_size: int = field(
        default=64,
        metadata={
            "help": "STS 任务中，每台 GPU 处理的 batch size。由于 STS 任务文本较短，通常使用较长的 batch size。"
        },
    )
    temperature: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "Similarity will be sim = sim / temperature before using them to compute loss."
            " A higher temperature can reduce the value of similarity between texts in downstream tasks."
        },
    )
    lora: bool = field(default=False, metadata={"help": "Use LoRA PEFT"})
    qlora: bool = field(default=False, metadata={"help": "Use QLoRA PEFT"})
    save_safetensors: bool = field(
        default=False, metadata={"help": "Save in safetensors format"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using distributed training, setting this to False can fix issues with gradient checkpointing."
        },
    )
