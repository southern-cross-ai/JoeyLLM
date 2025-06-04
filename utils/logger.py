import wandb

class WandBLogger:
    def __init__(self, project_name, name=None, config=None):
        wandb.init(project=project_name, name=name, config=config)

    def log_message(self, message):
        print(message)

    def log_metrics(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def watch_model(self, model, log="all", log_freq=10):
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        wandb.finish()


