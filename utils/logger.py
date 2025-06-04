import wandb

class WandBLogger:
    def __init__(self, project_name, config=None):
        """
        Initialize the W&B logger.
        :param project_name: Name of the W&B project.
        :param config: Optional dict for experiment config (like LR, batch size, etc.).
        """
        wandb.init(project=project_name, config=config)

    def log_message(self, message):
        """
        Log a message to console.
        W&B doesn’t have direct text logging, so this is for local console output.
        """
        print(message)

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B.
        :param metrics: Dict of metrics (e.g., {"train_loss": 0.1}).
        :param step: Optional step number for logs.
        """
        wandb.log(metrics, step=step)

    def watch_model(self, model):
        """
        Watch model parameters and gradients.
        """
        wandb.watch(model)

