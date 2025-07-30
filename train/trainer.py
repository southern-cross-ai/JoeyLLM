import torch
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from typing import Any
from torch.amp import GradScaler, autocast

class Trainer:
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        logger: Any,
        rank: int,
        world_size: int,
        total_steps: int,
        scheduler_cfg,
        accumulation_steps: int,
        save_model_path: str,
        log_freq: int,
        non_blocking: bool,
        device: torch.device = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}")
        
        # Move model to device, then wrap in DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank])

        self.optimizer = optimizer
        self.dataloader = dataloader
        self.logger = logger
        self.loss_fn = CrossEntropyLoss()
        self.global_step = 0
        self.accumulation_steps = accumulation_steps
        self.save_model_path = save_model_path
        self.log_freq = log_freq
        self.non_blocking = non_blocking

        # Mixed precision
        self.scaler = GradScaler()

        # Set up OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=scheduler_cfg.max_lr,
            total_steps=total_steps,
            pct_start=scheduler_cfg.pct_start,
            anneal_strategy=scheduler_cfg.anneal_strategy,
            div_factor=scheduler_cfg.div_factor,
            final_div_factor=scheduler_cfg.final_div_factor,
            cycle_momentum=scheduler_cfg.cycle_momentum,
            base_momentum=scheduler_cfg.base_momentum,
            max_momentum=scheduler_cfg.max_momentum,
            three_phase=scheduler_cfg.three_phase,
        )

        self.logger.print(f"🟢 Training Starting on rank {rank}")

    def save_model(self, path=None):
        path = path or self.save_model_path
        # Save only model weights for inference, overwrite each time
        if isinstance(self.model, DDP):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        if self.logger.is_main:
            self.logger.print(f"💾 Model saved to {path}")


    def epoch(self, epoch: int):
        self.model.train()

        for step, batch in enumerate(self.dataloader):

            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["target_ids"].to(self.device, non_blocking=True)
            
            if step % self.accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'): 
                outputs = self.model(inputs)
                
                loss = self.loss_fn(
                    outputs.view(-1, outputs.size(-1)),  # [B*T, V]
                    labels.view(-1)                      # [B*T]
            )
                
            lr = self.optimizer.param_groups[0]["lr"]
            
            if step % self.accumulation_steps == 0 and self.logger.is_main:
                self.logger.print(f"📝 Epoch {epoch} Step {step} Loss: {loss.item():.4f} LR: {lr:.6f}")
                self.logger.wb(
                    "log",
                    metrics={"train/loss": loss.item(), "train/lr": lr},
                    step=self.global_step
                )

            loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            self.global_step += 1

            if self.global_step % self.log_freq == 0 and self.logger.is_main:
                self.save_model()
                self.logger.print("Saving model!!!")

        # ✅ 🛠️ Final optimizer step for remaining gradients (AFTER loop)
        remaining = len(self.dataloader) % self.accumulation_steps
        if remaining != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()


    def train(self, epochs: int):
        for epoch in range(epochs):
            self.epoch(epoch)
