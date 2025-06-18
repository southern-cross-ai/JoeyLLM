import torch.distributed as dist
import wandb
from tqdm import tqdm

class Monitor:
    def __init__(self, project=None, run_name=None, config=None):
        self.rank = self._get_rank()
        self.is_main = self.rank == 0

        self.wandb_run = None
        if self.is_main:
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                reinit=True
            )

    def _get_rank(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def print(self, message: str):
        if self.is_main:
            print(message)

    def wb(self, command, **kwargs):
        if not self.is_main:
            return

        if command == "log":
            wandb.log(kwargs.get("data", {}), step=kwargs.get("step"))
        elif command == "start":
            model = kwargs.get("model")
            if model:
                wandb.watch(model, log=kwargs.get("log", "gradients"), log_freq=kwargs.get("log_freq", 100))
        elif command == "finish":
            wandb.finish()
        elif command == "save":
            path = kwargs.get("path")
            if path:
                wandb.save(path)
        else:
            raise ValueError(f"Unknown wandb command: {command}")

    def bar(self, mode, iterator, **kwargs):
        if not self.is_main:
            return iterator

        if mode == "epoch":
            return tqdm(iterator, leave=True, position=0, dynamic_ncols=True, **kwargs)
        elif mode == "step":
            return tqdm(iterator, leave=False, position=1, dynamic_ncols=True, **kwargs)
        else:
            return tqdm(iterator, **kwargs)
