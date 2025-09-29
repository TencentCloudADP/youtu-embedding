import os
import json
import random
import logging
import datasets
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from arguments import CustomTrainingArguments, DataArguments, ModelArguments
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, set_seed

from utils import dist_env
from model import TrainModel
from data import CustomCollator, CustomDataset
from sampler import MixedDatasetSampler, SingleDatasetSampler, DynamicBatchSizeSampler

try:
    from apex import amp
except ImportError:
    amp = None


logger = logging.getLogger(__name__)

def args_to_dtype(args):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def filter_too_long_instructions(tokenizer, dataset, query_max_len, passage_max_len):
    def filter_fn(example):
        if (len(example["query"][0]) > query_max_len * 10) or not (example["query"][1]):
            return False

        if (len(tokenizer.tokenize(example["query"][0].strip("\t\n :"))) >= query_max_len):
            return False

        for ex in example["positives"] + example["negatives"]:
            if (len(ex[0]) > passage_max_len * 10) or not (ex[1]):
                return False
            if (len(tokenizer.tokenize(ex[0].strip("\t\n :"))) >= passage_max_len):
                return False
        return True

    num_proc = 8
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)


def normalize_dataset_features(dataset):
    features = datasets.Features({
        "task_type": datasets.Value("string"),
        "query": datasets.Sequence(datasets.Value("string")),
        "positives": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        "negatives": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        "positive_scores": datasets.Sequence(datasets.Value("float32")),
        "negative_scores": datasets.Sequence(datasets.Value("float32"))
    })

    def convert_types(example):
        query = example.get("query", "")
        assert isinstance(query, list) and len(query) == 3, f"Expected query to be list or str, got {type(query)}"
        query = [str(item) for item in query]

        positives = example.get("positives", [["", "", "nothing"]])
        negatives = example.get("negatives", [["", "", "nothing"]])

        assert isinstance(positives, list)
        assert isinstance(negatives, list)

        if len(negatives) == 0:
            negatives = [["", "", "nothing"]]

        positive_scores = example.get("positive_scores", None)
        if len(positive_scores) == 0:
            positive_scores = [-1.0] * len(positives)
        assert isinstance(positive_scores, list)
        positive_scores = [float(score) for score in positive_scores]

        negative_scores = example.get("negative_scores", None)
        if len(negative_scores) == 0:
            negative_scores = [-1.0] * len(negatives)
        assert isinstance(negative_scores, list)
        negative_scores = [float(score) for score in negative_scores]

        return {
            "task_type": example.get("task_type", "ir"),
            "query": query,
            "positives": positives,
            "negatives": negatives,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores
        }

    return dataset.map(convert_types, features=features)


def process_dataset(data_args, training_args, num_samples, ds_name_to_samples, task_data):
    data_files = (
        [
            os.path.join(task_data, x)
            for x in os.listdir(task_data)
        ]
        if os.path.isdir(task_data)
        else [task_data]
    )

    data_files = [
        file for file in data_files if file.endswith("json") or file.endswith("jsonl")
    ]

    task_ds = []
    for i, file in enumerate(data_files, 1):
        with nullcontext():
            logger.info("Loading dataset(%s of %s) %s", i, len(data_files), file)
            tmp_ds = datasets.load_dataset("json", data_files=file, split="train")

            if "meta_data" in tmp_ds.column_names:
                tmp_ds = tmp_ds.remove_columns("meta_data")

            tmp_ds_len = len(tmp_ds)
            if tmp_ds_len > data_args.max_example_num_per_dataset:
                tmp_ds = tmp_ds.select(
                    random.sample(
                        list(range(tmp_ds_len)), data_args.max_example_num_per_dataset
                    )
                )

            if "query" in tmp_ds.features:
                if isinstance(tmp_ds[0]["query"], (tuple, list)):
                    if training_args.skip_filter_too_long_instruction:
                        logger.info(f"skip filtering too long instruction for {file}")
                    else:
                        logger.info(f"Filtering out embedding samples with too long instructions for {file}")
                        tmp_ds = filter_too_long_instructions(tokenizer, tmp_ds, data_args.query_max_len, data_args.passage_max_len)

                    if num_samples:
                        assert (file.split("/")[-1] in num_samples), f'Missing num_samples for {file.split("/")[-1]}'

                        tmp_ds_len = len(tmp_ds)
                        samples = num_samples[file.split("/")[-1]]

                        if tmp_ds_len > samples:
                            tmp_ds = tmp_ds.select(
                                random.sample(list(range(tmp_ds_len)), samples)
                            )

                tmp_ds = normalize_dataset_features(tmp_ds)
                ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
                logger.info(f"dataset size for {file}: {len(tmp_ds)}")
                task_ds.append(tmp_ds)
                continue

            logger.info("Skipping dataset %s as its type could not be identified", file)

    return task_ds, ds_name_to_samples


def load_ir_train_data(data_args, training_args, num_samples, ds_name_to_samples):
    if data_args.ir_train_data:
        ir_ds, ds_name_to_samples = process_dataset(data_args, training_args, num_samples, ds_name_to_samples, data_args.ir_train_data)
        ir_lens = [len(t) for t in ir_ds]

        ir_ds = datasets.concatenate_datasets(ir_ds)
        logger.info("Embedding mode IR: %d samples", len(ir_ds))
    else:
        ir_ds = None
        ir_lens = []
        logger.info("Embedding mode IR: 0 samples (no data provided)")
    
    return ir_ds, ir_lens, ds_name_to_samples


def load_sts_train_data(data_args, training_args, num_samples, ds_name_to_samples):
    if data_args.sts_train_data:
        sts_ds, ds_name_to_samples = process_dataset(data_args, training_args, num_samples, ds_name_to_samples, data_args.sts_train_data)
        sts_lens = [len(t) for t in sts_ds]

        sts_ds = datasets.concatenate_datasets(sts_ds)
        logger.info("Embedding mode STS: %d samples", len(sts_ds))
    else:
        sts_ds = None
        sts_lens = []
        logger.info("Embedding mode STS: 0 samples (no data provided)")

    return sts_ds, sts_lens, ds_name_to_samples


def load_train_data(data_args, training_args):
    num_samples = None
    ds_name_to_samples = {}

    ds_types, ds_embedding_lens = [], []
    assert data_args.ir_train_data or data_args.sts_train_data, "At least one of ir_train_data or sts_train_data must be provided"

    if data_args.data_sampler == "dynamic":
        assert data_args.ir_train_data and data_args.sts_train_data,  \
            "Dynamic sampler requires both ir_train_data and sts_train_data to be provided."

    ir_ds, ir_lens, ds_name_to_samples = load_ir_train_data(data_args, training_args, num_samples, ds_name_to_samples)
    sts_ds, sts_lens, ds_name_to_samples = load_sts_train_data(data_args, training_args, num_samples, ds_name_to_samples)

    ds_embedding_lens.extend(ir_lens)
    ds_types.extend(["ir"] * len(ir_lens))

    ds_embedding_lens.extend(sts_lens)
    ds_types.extend(["sts"] * len(sts_lens))

    ds = [ds for ds in [ir_ds, sts_ds] if ds is not None]
    logger.info("Dataset type distribution: IR (%d sub-datasets), STS (%d sub-datasets)", len(ir_lens), len(sts_lens))

    os.makedirs(training_args.output_dir, exist_ok=True)

    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w") as f:
        json.dump(ds_name_to_samples, f)
    os.system("pkill -9 -f spawn")
    return ds, ds_types, ds_embedding_lens


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_only_model = True

    assert data_args.data_sampler in ["single", "mixed", "dynamic"], "data_sampler must be 'single', 'mixed', or 'dynamic'."

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        trust_remote_code=True,
    )
    logger.info("Config: %s", config)

    ds, ds_types, ds_embedding_lens = load_train_data(data_args, training_args)

    model = TrainModel(
        model_name_or_path=model_args.model_name_or_path,
        pooling_method=model_args.pooling_method,
        normalized=model_args.normalized,
        projection=model_args.projection,
        attn=model_args.attn,
        temperature=training_args.temperature,
        ir_negatives_cross_device=training_args.ir_negatives_cross_device,
        sts_negatives_cross_device=training_args.sts_negatives_cross_device,
        multi_layer_loss=model_args.multi_layer_loss,
        positive_group_size=data_args.positive_group_size,
        negative_group_size=data_args.negative_group_size,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=args_to_dtype(training_args),
        use_cache=False,
    )

    train_dataset = CustomDataset(
        dataset=ds,
        args=data_args,
        tokenizer=tokenizer,
        max_seq_len=max(
            data_args.query_max_len,
            data_args.passage_max_len,
        ),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": CustomCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            model_name_or_path=model_args.model_name_or_path,
        ),
        "tokenizer": tokenizer,
    }
    trainer = Trainer(**trainer_kwargs)

    if len(ds_embedding_lens) > 1:
        assert training_args.dataloader_drop_last, "Multiple datasets are only supported with dropping the last incomplete batch, set `--dataloader_drop_last`"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        global_batch_size_for_chunking = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

        if dist.is_initialized() and dist.is_available():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            global_batch_size_for_chunking = global_batch_size_for_chunking * dist.get_world_size()
        else:
            num_replicas = 1
            rank = 0

        def get_train_dataloader_with_single_sampler(self):
            train_sampler = SingleDatasetSampler(
                dataset=self.train_dataset,
                ds_lens=ds_embedding_lens,
                num_replicas=num_replicas,
                rank=rank,
                global_batch_size_for_chunking=global_batch_size_for_chunking,
                seed=training_args.seed
            )

            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

            def set_epoch(epoch):
                if hasattr(train_sampler, 'set_epoch'):
                    train_sampler.set_epoch(epoch)
                    if dist.get_rank() == 0:
                        print(f"🔥 DataLoader.set_epoch({epoch}) called, passed to sampler")

            train_dataloader.set_epoch = set_epoch

            return train_dataloader

        def get_train_dataloader_with_mixed_sampler(self):
            train_sampler = MixedDatasetSampler(
                dataset=self.train_dataset,
                ds_lens=ds_embedding_lens,
                num_replicas=num_replicas,
                rank=rank,
                per_device_batch_size=training_args.per_device_train_batch_size,
                seed=training_args.seed,
                epoch=0
            )

            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

            def set_epoch(epoch):
                if hasattr(train_sampler, 'set_epoch'):
                    train_sampler.set_epoch(epoch)
                    if dist.get_rank() == 0:
                        print(f"🔥 MixedDatasetSampler.set_epoch({epoch}) called")

            train_dataloader.set_epoch = set_epoch

            return train_dataloader

        def get_train_dataloader_with_dynamic_batch(self):
            train_sampler = DynamicBatchSizeSampler(
                dataset=self.train_dataset,
                ds_lens=ds_embedding_lens,
                ds_types=ds_types,
                num_replicas=num_replicas,
                rank=rank,
                ir_per_device_batch_size=training_args.ir_per_device_batch_size,
                sts_per_device_batch_size=training_args.sts_per_device_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                seed=training_args.seed
            )

            class DynamicBatchDataLoader:
                def __init__(self, dataset, sampler, collate_fn):
                    self._length = None
                    self.dataset = dataset
                    self.sampler = sampler
                    self.collate_fn = collate_fn

                def __iter__(self):
                    generator = torch.Generator()
                    generator.manual_seed(self.sampler.seed + self.sampler.epoch)

                    current_offset = 0
                    all_global_batches = []

                    for batch_info in self.sampler.dataset_batches:
                        ds_len = batch_info['dataset_length']
                        num_batches = batch_info['num_batches']
                        global_batch_size = batch_info['global_batch_size']

                        if num_batches == 0:
                            current_offset += ds_len
                            continue

                        indices = torch.randperm(ds_len, generator=generator) + current_offset

                        for i in range(num_batches):
                            start_idx = i * global_batch_size
                            end_idx = start_idx + global_batch_size
                            global_batch_indices = indices[start_idx : end_idx].tolist()

                            all_global_batches.append({
                                'global_indices': global_batch_indices,
                                'dataset_type': batch_info['dataset_type'],
                                'per_device_batch_size': batch_info['per_device_batch_size'],
                                'dataset_idx': batch_info['dataset_idx']
                            })

                        current_offset += ds_len

                    batch_order = torch.randperm(len(all_global_batches), generator=generator).tolist()
                    shuffled_global_batches = [all_global_batches[i] for i in batch_order]

                    for i in range(self.sampler.iterations_per_gpu):
                        if i >= len(shuffled_global_batches):
                            break

                        global_batch = shuffled_global_batches[i]
                        per_device_batch_size = global_batch['per_device_batch_size']

                        start_idx = rank * per_device_batch_size
                        end_idx = start_idx + per_device_batch_size
                        gpu_indices = global_batch['global_indices'][start_idx : end_idx]

                        batch_samples = [self.dataset[idx] for idx in gpu_indices]
                        batch = self.collate_fn(batch_samples)
                        yield batch

                def __len__(self):
                    if self._length is None:
                        self._length = self.sampler.iterations_per_gpu
                    return self._length

                def set_epoch(self, epoch):
                    if hasattr(self.sampler, 'set_epoch'):
                        self.sampler.set_epoch(epoch)

            train_dataloader = DynamicBatchDataLoader(
                dataset=self.train_dataset,
                sampler=train_sampler,
                collate_fn=self.data_collator
            )

            return train_dataloader

        sampler_dict = {
            "single": get_train_dataloader_with_single_sampler.__get__(trainer),
            "mixed": get_train_dataloader_with_mixed_sampler.__get__(trainer),
            "dynamic": get_train_dataloader_with_dynamic_batch.__get__(trainer)
        }
        trainer.get_train_dataloader = sampler_dict[data_args.data_sampler]

    global_step = 0
    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        nonlocal global_step
        global_step += 1

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    trainer.training_step = training_step.__get__(trainer)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if dist_env.is_main_process:
        config.to_json_file(training_args.output_dir + "/config.json")
        path = Path(model_args.model_name_or_path).joinpath("*.py")
        os.system(f"cp {str(path)} {training_args.output_dir}")

    logger.info("Starting training")
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
