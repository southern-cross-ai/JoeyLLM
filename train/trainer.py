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
        self.accumulation_steps = 4

        # Mixed precision
        self.scaler = GradScaler()

        # Set up OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.01,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            three_phase=False,
        )

        self.logger.print(f"🟢 Training Starting on rank {rank}")

    def epoch(self, epoch: int):
        self.model.train()

        for step, batch in enumerate(self.dataloader):
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            if step % self.accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'): 
                outputs = self.model(inputs)
                
                loss = self.loss_fn(
                    outputs.view(-1, outputs.size(-1)),  # [B*T, V]
                    labels.view(-1)                      # [B*T]
            )
                
            lr = self.optimizer.param_groups[0]["lr"]
            
            if step % 10 == 0 and self.logger.is_main:
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

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.epoch(epoch)
