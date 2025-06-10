import os
import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch distributed if environment variables are set."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank, True


def cleanup_distributed() -> None:
    """Destroy distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()