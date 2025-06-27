# from tqdm import tqdm

import os
class Monitor:
    def __init__(
        self, 
        wandb_mode="online", 
        project="JoeyLLM", 
        run_name=None, 
        ):
        
        # Set Rank
        self.rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_main = self.rank == 0

        
        # Set WandB
        self.wandb_mode = wandb_mode.lower()
        self.project = project
        self.run_name = run_name
        self.wandb_run = None 
        if self.is_main:
            os.environ["WANDB_MODE"] = self.wandb_mode

    def wb(self, command, **kwargs):
        if self.is_main and self.wandb_mode != "disabled":
            import wandb             
            
            if command == "Start": 
                self.wandb_run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=kwargs.get("config"),
                    resume="allow",  # or "must" or "never"
                ) 
            
            elif command == "model":
                if kwargs.get("model") is None:
                    raise ValueError("WandB is missing the model object")
                wandb.watch(kwargs["model"], 
                            log=kwargs.get("log", "gradients"), 
                            log_freq=kwargs.get("log_freq", 1000))
            
            elif command == "log":
                if not isinstance(kwargs.get("metrics"), dict):
                    raise ValueError("Did not get the logs, please check")
                wandb.log(kwargs.get("metrics"), step=kwargs.get("step"))

            elif command == "Stop":
                wandb.finish()

            else:
                raise ValueError(f"Unknown wandb command: {command}")

    def print(self, message: str):
        if self.is_main:
            print(message)
    





#     # def bar(self, mode, iterator, **kwargs):
#     #     if not self.is_main:
#     #         return iterator

#     #     if mode == "epoch":
#     #         return tqdm(iterator, leave=True, position=0, dynamic_ncols=True, **kwargs)
#     #     elif mode == "step":
#     #         return tqdm(iterator, leave=False, position=1, dynamic_ncols=True, **kwargs)
#     #     else:
#     #         return tqdm(iterator, **kwargs)
