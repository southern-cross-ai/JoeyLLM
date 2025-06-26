import torch
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from typing import Any


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

        # Set up OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.03,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            three_phase=False
        )

        self.logger.print(f"🟢 Training Starting on rank {rank}")

    def epoch(self, epoch: int):
        self.model.train()

        for step, batch in enumerate(self.dataloader):
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            outputs = self.model(inputs)

            loss = self.loss_fn(
                outputs.view(-1, outputs.size(-1)),  # [B*T, V]
                labels.view(-1)                      # [B*T]
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 🔑 Scheduler updates every batch

            if step % 10 == 0:
                self.logger.print(f"📝 Epoch [{epoch}] Step [{step}] Loss: {loss.item():.4f}")

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.epoch(epoch)
