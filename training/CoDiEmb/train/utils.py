import os
import time
import torch.distributed as dist
from contextlib import contextmanager


class DistributedEnv:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    def wait_for_everyone(self):
        dist.barrier()

    def _do_it_first(self, first):
        if not first:
            self.wait_for_everyone()
        yield

        if first:
            self.wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        yield from self._do_it_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        yield from self._do_it_first(self.is_local_main_process)


dist_env = DistributedEnv()


def broadcast_object_list(object_list):
    if dist_env.world_size == 1:
        return object_list

    dist.broadcast_object_list(object_list, src=0)
    return object_list


def broadcast_object(object_):
    object_list = [object_]
    return broadcast_object_list(object_list)[0]


def sequential_execution(closure, time_to_wait=-1):
    from shiny.utilities.logger import logger

    def wrappeed_callable():
        logger.info(f"rank-{dist_env.local_rank} start execution")
        res = closure()
        logger.info(f"rank-{dist_env.local_rank} end execution")
        return res

    def one_by_one():
        if not dist.is_initialized():
            raise RuntimeError("must be used after distributed initialized")
        global_rank = int(os.environ("RANK", "0"))
        node_idx = global_rank // dist_env.local_world_size
        start_rank = node_idx * dist_env.local_world_size
        end_rank = (node_idx + 1) * dist_env.local_world_size
        ranks = list(range(start_rank, end_rank))
        pg = dist.new_group(ranks=ranks)
        for i in range(dist_env.local_world_size):
            if dist_env.local_rank == i:
                res = wrappeed_callable()
            dist.barrier(pg)
        return res

    def overlap():
        wait_time = time_to_wait * dist_env.local_rank
        time.sleep(wait_time)
        return wrappeed_callable()

    if time_to_wait == -1:
        return one_by_one()
    else:
        return overlap()
