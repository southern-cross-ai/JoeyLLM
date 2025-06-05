from datetime import datetime
import wandb
import os

# os.environ["WANDB_SILENT"] = "true"

class wandbLogger:

    @staticmethod
    def set_mode(mode="offline"):
        """
        Set the wandb logging mode.
        :param mode: "online", "offline", or "disabled"
        """
        os.environ["WANDB_MODE"] = mode
    
    def __init__(self, project_name, name=None, config=None):
        if name is None:
            name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=project_name, name=name, config=config)

    def log_message(self, message):
        print(message)

    def log_metrics(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def watch_model(self, model, log="all", log_freq=10):
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        wandb.finish()


