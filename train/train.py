import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


class JoeyLLMTrainer:
    """
    Trainer class for training JoeyLLM models using PyTorch with support for:
    - Single GPU training
    - Gradient accumulation
    - Checkpoint saving and resuming
    - Learning rate scheduling
    - WandB experiment tracking
    """

    def __init__(self, cfg, model, train_loader, val_loader=None):
        """
        Initializes training environment.

        Args:
            cfg (DictConfig): Configuration object loaded via Hydra.
            model (nn.Module): JoeyLLM model instance.
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader, optional): DataLoader for validation set.
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Determine device
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)

        # Loss function: Cross-entropy over token logits
        self.criterion = nn.CrossEntropyLoss()

        # Epoch and step counters
        self.epoch = 0
        self.step = 0

        # Gradient accumulation setup
        self.accum_steps = cfg.train.gradient_accumulation_steps

        # WandB setup
        self.use_wandb = cfg.train.wandb.log
        if self.use_wandb:
            wandb.init(
                # project=cfg.train.wandb.project,
                name=f"train-{datetime.now().strftime('%d%m%Y-%H%M%S')}",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
            wandb.watch(model)

        # Load from checkpoint if available
        self.load_checkpoint()

    def save_checkpoint(self):
        """
        Saves current model, optimizer, and scheduler states to a checkpoint file.
        """
        os.makedirs(self.cfg.train.checkpoint_path, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.cfg
        }

        path = os.path.join(self.cfg.train.checkpoint_path, f"joeyllm_epoch_{self.epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint at epoch {self.epoch}")

    def load_checkpoint(self):
        """
        Loads latest checkpoint if one exists in the specified checkpoint directory.
        Resumes model training from that checkpoint.
        """
        resume_path = self.cfg.train.resume_from or self.cfg.train.checkpoint_path
        if not os.path.exists(resume_path):
            return

        try:
            checkpoints = [f for f in os.listdir(resume_path) if f.endswith(".pt")]
            if not checkpoints:
                return

            # Extract latest checkpoint by epoch number
            latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = torch.load(os.path.join(resume_path, latest))

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.epoch = checkpoint['epoch'] + 1
            self.step = checkpoint['step']

            print(f"Resumed training from epoch {self.epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def train_single_gpu(self):
        """
        Main training loop (for single gpu) over epochs and batches with:
        - Gradient accumulation
        - Periodic checkpoint saving
        - Loss logging
        """
        for epoch in range(self.epoch, self.cfg.train.epochs):
            self.model.train()
            epoch_loss = 0.0
            self.epoch = epoch

            # Progress bar setup
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}")

            self.optimizer.zero_grad()

            for i, batch in pbar:
                # Move inputs to the correct device
                input_ids = batch["input_ids"].to(self.device)

                outputs = self.model(input_ids[:, :-1])
                targets = input_ids[:, 1:]

                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )   


                # Normalize loss by gradient accumulation steps
                loss = loss / self.accum_steps
                loss.backward()
                epoch_loss += loss.item()
                self.step += 1

                # Perform optimizer step after accumulating gradients
                if (i + 1) % self.accum_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix(loss=loss.item() * self.accum_steps)

                # Log to WandB
                if self.use_wandb:
                    wandb.log({"loss": loss.item() * self.accum_steps, "epoch": epoch})

            print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.cfg.train.save_every == 0:
                self.save_checkpoint()

        if self.use_wandb:
            wandb.finish()
