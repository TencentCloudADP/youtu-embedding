import torch
import logging
from dataclasses import dataclass
from typing import Iterator, List

logger = logging.getLogger(__name__)


@dataclass
class MixedDatasetSampler(torch.utils.data.Sampler[int]):
    """
    Mixed Dataset Sampler

    Core concept:
    - Within each GPU: samples in one batch must come from the same dataset
    - Across different GPUs: can simultaneously process different datasets
    - Key fix: ensure all GPUs have exactly the same number of training steps

    Implementation logic:
    1. Calculate the total number of complete batches available globally
    2. Ensure all GPUs process the same number of batches
    3. Within each GPU, ensure single batch comes from the same dataset
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        ds_lens: List[int],
        num_replicas: int,
        rank: int,
        per_device_batch_size: int,
        seed: int = 0,
        epoch: int = 0,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer.")
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}].")
        if per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be a positive integer.")

        self.dataset = dataset
        self.ds_lens = ds_lens
        self.per_device_batch_size = per_device_batch_size

        self.rank = rank
        self.num_replicas = num_replicas
    
        self.seed = seed
        self.epoch = epoch

        self.batches_per_dataset = [length // per_device_batch_size for length in ds_lens]
        self.total_batches_all_datasets = sum(self.batches_per_dataset)

        self.batches_per_gpu = self.total_batches_all_datasets // num_replicas
        self.total_batches_used = self.batches_per_gpu * num_replicas
        self.num_samples = self.batches_per_gpu * per_device_batch_size

        if self.rank == 0:
            print(f"=== MixedDatasetSampler Debug Info ===")
            total_samples = sum(self.ds_lens)
            used_samples = self.total_batches_used * per_device_batch_size
            discarded_samples = total_samples - used_samples
            discarded_batches = self.total_batches_all_datasets - self.total_batches_used

            print(f"Total samples: {total_samples}")
            print(f"Used samples: {used_samples}")
            print(f"Discarded samples: {discarded_samples}")
            print(f"Discard ratio: {discarded_samples / total_samples * 100:.2f}%")
            print(f"Total batches: {self.total_batches_all_datasets}")
            print(f"Used batches: {self.total_batches_used}")
            print(f"Discarded batches: {discarded_batches}")
            print(f"Batches per GPU: {self.batches_per_gpu}")
            print(f"Samples per GPU: {self.num_samples}")
            print(f"=== MixedDatasetSampler Debug Info ===")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_all_dataset_batches(self, generator):
        all_dataset_batches = []
        current_offset = 0

        for dataset_idx, (ds_len, num_batches) in enumerate(zip(self.ds_lens, self.batches_per_dataset)):
            if num_batches == 0:
                current_offset += ds_len
                continue

            indices = torch.randperm(ds_len, generator=generator) + current_offset

            full_batches = list(torch.split(indices[:num_batches * self.per_device_batch_size], self.per_device_batch_size))

            for batch in full_batches:
                all_dataset_batches.append((dataset_idx, batch.tolist()))

            current_offset += ds_len

        return all_dataset_batches

    def __iter__(self) -> Iterator[int]:
        gpu_emojis = ["🔥", "⚡", "🌟", "💎", "🚀", "🎯", "⭐", "🌈"]
        gpu_emoji = gpu_emojis[self.rank % len(gpu_emojis)]
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self.rank * 10000)

        print(f"{gpu_emoji} [GPU {self.rank}] 🎲 Using random seed: {self.seed + self.epoch + self.rank * 10000}")

        all_dataset_batches = self.get_all_dataset_batches(generator)

        batch_order = torch.randperm(len(all_dataset_batches), generator=generator).tolist()
        shuffled_batches = [all_dataset_batches[i] for i in batch_order]
        shuffled_batches = shuffled_batches[:self.total_batches_used]

        gpu_batches = []
        start_idx = self.rank
        for i in range(self.batches_per_gpu):
            batch_idx = (start_idx + i * self.num_replicas) % len(shuffled_batches)
            gpu_batches.append(shuffled_batches[batch_idx])

        final_indices = []
        dataset_distribution = {}

        for dataset_idx, batch_indices in gpu_batches:
            if dataset_idx not in dataset_distribution:
                dataset_distribution[dataset_idx] = 0
            dataset_distribution[dataset_idx] += 1
            final_indices.extend(batch_indices)

        print(f"{gpu_emoji} [GPU {self.rank}] 🎯 Dataset distribution: {dataset_distribution}")
        print(f"{gpu_emoji} [GPU {self.rank}] 📏 Processing {len(final_indices)} samples (batches: {len(gpu_batches)})")

        yield from final_indices

    def __len__(self) -> int:
        return self.num_samples


class SingleDatasetSampler(torch.utils.data.Sampler[int]):
    """
    Sampler used when training on multiple datasets to ensure each
    batch only contains samples from one dataset,
    discarding any leftover samples that don't fit into a full batch.
    Handles distributed training and ensures consistent shuffling across epochs and ranks.
    """

    # Fields from old class that are not direct parameters anymore:
    # _num_samples: int = None -> Not needed, len is calculated differently
    # data_source: CustomDataset = None -> Renamed to dataset, type is torch.utils.data.Dataset
    # replacement: bool = False -> Not used by this sampler's logic

    def __init__(
        self,
        dataset: torch.utils.data.Dataset, # Full dataset
        ds_lens: List[int], # Lengths of sub-datasets
        num_replicas: int,
        rank: int,
        global_batch_size_for_chunking: int, # Global batch size for dataset chunking
        seed: int = 0,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer.")
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval"
                f" [0, {num_replicas - 1}]."
            )
        if global_batch_size_for_chunking <= 0:
            raise ValueError("global_batch_size_for_chunking must be a positive integer.")

        self.epoch = 0
        self.seed = seed
        self.rank = rank
        self.dataset = dataset
        self.ds_lens = ds_lens
        self.num_replicas = num_replicas
        self.global_batch_size_for_chunking = global_batch_size_for_chunking

        num_total_samples_from_core_logic = 0

        for length in self.ds_lens:
            full_batches = length // self.global_batch_size_for_chunking
            num_total_samples_from_core_logic += full_batches * self.global_batch_size_for_chunking

        self.num_total_samples_from_core_logic = num_total_samples_from_core_logic
        self.num_samples_per_replica = self.num_total_samples_from_core_logic // self.num_replicas

        # self.total_size will be equal to self.num_total_samples_from_core_logic
        self.total_size = self.num_samples_per_replica * self.num_replicas

        if self.rank == 0:
            print(f"=== StrictSingleDatasetSampler Debug Info ===")
            
            total_samples = sum(self.ds_lens)
            total_discarded = total_samples - self.num_total_samples_from_core_logic
            
            print(f"Total samples: {total_samples}")
            print(f"Discarded samples: {total_discarded}")
            print(f"Discard ratio: {total_discarded / total_samples * 100:.2f}%")

            print(f"Samples per GPU per epoch: {self.num_samples_per_replica}")
            print(f"If per_device_batch_size = {self.global_batch_size_for_chunking // self.num_replicas}:")
            print("=== End Debug Info ===")

    def get_chunks(self, global_idxs, ds_indices_per_dataset_chunks):
        # Split into chunks by global_batch_size_for_chunking, discard the part that doesn't fit into batch_size
        # This is the primary "drop_last" mechanism of this sampler.
        chunked = list(torch.split(global_idxs, self.global_batch_size_for_chunking))
        if len(chunked) > 0 and len(chunked[-1]) < self.global_batch_size_for_chunking:
            chunked.pop()  # Remove the last incomplete batch

        ds_indices_per_dataset_chunks.extend(c for c in chunked if len(c) == self.global_batch_size_for_chunking)
        return ds_indices_per_dataset_chunks

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        offset = 0
        ds_indices_per_dataset_chunks = []  # List of tensors, each tensor is a chunk/batch

        for length in self.ds_lens:
            assert length > 0
            # shuffle [0 .. length - 1]
            idxs = torch.randperm(length, generator=g)

            # convert local idx -> global idx (offset)
            global_idxs = idxs + offset
            offset += length

            ds_indices_per_dataset_chunks = self.get_chunks(global_idxs, ds_indices_per_dataset_chunks)

        assert len(ds_indices_per_dataset_chunks) > 0, \
            "=== StrictSingleDatasetSampler: No full batches could be formed. Returning empty iterator. ==="

        stacked_chunks = torch.stack(ds_indices_per_dataset_chunks, dim=0)
        order = torch.randperm(stacked_chunks.size(0), generator=g)
        shuffled_stacked_chunks = stacked_chunks[order]
        all_indices = shuffled_stacked_chunks.view(-1).tolist()

        # The self.drop_last parameter (from training_args) does not necessitate
        # further truncation or padding of all_indices here because self.total_size
        # is already len(all_indices) due to the way num_total_samples_from_core_logic
        # and global_batch_size_for_chunking are defined.

        indices_for_this_rank = all_indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices_for_this_rank)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True` (which is implicit in this sampler's design),
        this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DynamicBatchSizeSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        ds_lens: List[int],
        ds_types: List[str],  # ["ir", "sts", "ir", ...] identifies the type of each sub-dataset
        num_replicas: int,
        rank: int,
        ir_per_device_batch_size: int,
        sts_per_device_batch_size: int,
        gradient_accumulation_steps: int = 1,
        seed: int = 0,
    ):
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}].")
        if len(ds_lens) != len(ds_types):
            raise ValueError(f"ds_lens length ({len(ds_lens)}) must match ds_types length ({len(ds_types)})")

        self.epoch = 0
        self.seed = seed
        self.rank = rank
        self.num_replicas = num_replicas

        self.dataset = dataset
        self.ds_lens = ds_lens
        self.ds_types = ds_types

        self.ir_per_device_batch_size = ir_per_device_batch_size
        self.sts_per_device_batch_size = sts_per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.ir_global_batch_size = ir_per_device_batch_size * num_replicas * gradient_accumulation_steps
        self.sts_global_batch_size = sts_per_device_batch_size * num_replicas * gradient_accumulation_steps

        self.dataset_batches = []
        for i, (ds_len, ds_type) in enumerate(zip(ds_lens, ds_types)):
            if ds_type == "ir":
                global_batch_size = self.ir_global_batch_size
                per_device_batch_size = ir_per_device_batch_size
            elif ds_type == "sts":
                global_batch_size = self.sts_global_batch_size
                per_device_batch_size = sts_per_device_batch_size
            else:
                raise ValueError(f"Unknown dataset type: {ds_type}")

            num_batches = ds_len // global_batch_size

            self.dataset_batches.append({
                'dataset_idx': i,
                'dataset_type': ds_type,
                'dataset_length': ds_len,
                'global_batch_size': global_batch_size,
                'per_device_batch_size': per_device_batch_size,
                'num_batches': num_batches,
                'samples_used': num_batches * global_batch_size
            })

        self.total_iterations = sum(batch_info['num_batches'] for batch_info in self.dataset_batches)
        self.iterations_per_gpu = self.total_iterations

        if self.rank == 0:
            self._print_debug_info()

    def _print_debug_info(self):
        print(f"\n{'='*80}")
        print(f"🎯 DynamicBatchSizeSampler Configuration")
        print(f"Number of GPUs: {self.num_replicas}")
        print(f"IR per device batch size: {self.ir_per_device_batch_size}")
        print(f"STS per device batch size: {self.sts_per_device_batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"IR global batch size: {self.ir_global_batch_size}")
        print(f"STS global batch size: {self.sts_global_batch_size}")
        print(f"\n📊 Sub-dataset statistics:")

        total_used = 0
        total_samples = 0
        for i, batch_info in enumerate(self.dataset_batches):
            total_samples += batch_info['dataset_length']
            total_used += batch_info['samples_used']

        print(f"Total samples: {total_samples}")
        print(f"Used samples: {total_used}")
        print(f"Discarded samples: {total_samples - total_used}")
        print(f"Discard ratio: {(total_samples - total_used) / total_samples * 100:.2f}%")
        print(f"Total iterations: {self.total_iterations}")
        print(f"Iterations per GPU: {self.iterations_per_gpu}")
        print(f"{'='*80}\n")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """
        Returns a virtual index sequence with length equal to the current GPU's iteration count
        Ensures compatibility with Trainer
        """
        return iter(range(self.iterations_per_gpu))

    def __len__(self) -> int:
        """Returns the current GPU's iteration count"""
        return self.iterations_per_gpu
