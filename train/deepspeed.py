import os
from tqdm import tqdm
import deepspeed
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsnapshot import Snapshot, Stateful


class DeepSpeedTrainer(Stateful):
    """
    Trainer class for distributed training using DeepSpeed.

    Features:
    - ZeRO-powered memory-efficient training
    - Automatic mixed precision support
    - DeepSpeed checkpointing for large-scale models
    - Simple integration with PyTorch DataLoader
    """

    def __init__(self, model, dataloader: DataLoader, logger, config_path="deepspeed_config.json", rank: int = 0):
        """
        Initialize the trainer with model, dataloader, logger, and DeepSpeed config.

        Args:
            model (nn.Module): The PyTorch model to train.
            dataloader (DataLoader): Training data loader.
            logger: Logger object to track metrics.
            config_path (str): Path to DeepSpeed config JSON.
            rank (int): Rank ID for distributed training.
        """
        self.rank = rank
        self.dataloader = dataloader
        self.logger = logger
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        self.global_step = 0
        self.snapshot_dir = "snapshots"
        self.snapshot_app_state = {"trainer": self}

        self.model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config_path
        )

        self.criterion = nn.CrossEntropyLoss()

    def fit(self, num_epochs, resume_from_latest=True):
        """
        Run training loop.

        Args:
            num_epochs (int): Number of training epochs.
            resume_from_latest (bool): Whether to resume from the latest snapshot.

        Returns:
            float: Final epoch average loss.
        """
        if resume_from_latest:
            self._resume_latest_snapshot()

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            self._save_checkpoint(epoch)

        if self.logger:
            self.logger.log_metrics({
                "final_epoch": num_epochs,
                "final_loss": train_loss
            }, step=self.global_step)

        return train_loss

    def _train_epoch(self, epoch):
        """
        Run one training epoch.

        Args:
            epoch (int): Current epoch index.

        Returns:
            float: Average running loss.
        """
        self.model_engine.train()
        running_loss = 0.0
        total_batches = 0

        progress = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False) if self.rank == 0 else self.dataloader

        for batch_idx, batch in enumerate(progress):
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model_engine(inputs)
            loss = self.compute_loss(outputs, labels)

            self.model_engine.backward(loss)
            self.model_engine.step()

            loss_value = loss.item()
            running_loss += loss_value
            total_batches += 1
            avg_loss = running_loss / total_batches

            if self.rank == 0:
                progress.set_description(f"Epoch {epoch} | Batch {batch_idx}")
                progress.set_postfix(loss=loss_value, avg=avg_loss)

            if self.logger:
                self.logger.log_metrics({
                    "train_loss": loss_value,
                    "avg_running_loss": avg_loss
                }, step=self.global_step)

            self.global_step += 1

        if self.rank == 0:
            progress.close()
            tqdm.write(f"✅ Epoch {epoch} | Final Avg Running Loss: {avg_loss:.4f}")

        return avg_loss

    def compute_loss(self, outputs, labels):
        """Compute CrossEntropy loss for reshaped outputs and labels."""
        B, T, V = outputs.size()
        return self.criterion(outputs.view(B * T, V), labels.view(B * T))

    def _save_checkpoint(self, epoch):
        """Save DeepSpeed checkpoint for the given epoch."""
        if self.rank == 0:
            checkpoint_dir = os.path.join(self.snapshot_dir, f"epoch_{epoch}")
            self.model_engine.save_checkpoint(checkpoint_dir)
            tqdm.write(f"💾 Saved DeepSpeed checkpoint at {checkpoint_dir}")

    def _resume_latest_snapshot(self):
        """Resume training from the latest DeepSpeed checkpoint."""
        latest_path = os.path.join(self.snapshot_dir, "latest")
        if not os.path.exists(latest_path):
            if self.rank == 0:
                print("🚫 No snapshot found at 'latest'.")
            return False

        self.model_engine.load_checkpoint(latest_path)
        self.model_engine.train()
        return True

    def state_dict(self):
        """Return trainer state dictionary."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state):
        """Load trainer state dictionary."""
        self.global_step = state.get("global_step", 0)
