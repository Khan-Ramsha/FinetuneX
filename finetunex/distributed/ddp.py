import os
import torch
from torch.distributed import init_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)

    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )