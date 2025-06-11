import torch
from torch.amp import autocast, GradScaler
from torchsnapshot import Snapshot
import torch.distributed as dist
from tqdm import tqdm
import os
import glob

class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        logger,
        scheduler=None,
        device="cuda",
        rank: int = 0
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.device = device
        self.scaler = GradScaler(device=self.device)
        self.rank = rank
        self.global_step = 0
        self.save_interval = 2000  


        self.snapshot_dir = "snapshots"
        self.snapshot_app_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
        }
        if self.scheduler:
            self.snapshot_app_state["scheduler"] = self.scheduler

    def compute_loss(self, outputs, labels):
        B, T, V = outputs.size()
        outputs = outputs.view(B * T, V)
        labels = labels.view(B * T)
        return self.criterion(outputs, labels)

    def _train_epoch(self, epoch):
        self.model.train()
        self.running_loss = 0.0
        self.total_batches = 0

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False) if self.rank == 0 else self.dataloader

        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = loss.item()
            self.running_loss += loss_value
            self.total_batches += 1
            avg_running_loss = self.running_loss / self.total_batches

            # Fixed-step snapshotting
            if self.global_step % self.save_interval == 0:
                snapshot_path = os.path.join(self.snapshot_dir, "latest")

                if dist.is_initialized():
                    dist.barrier()
                Snapshot.take(
                    path=snapshot_path,
                    app_state=self.snapshot_app_state,
                    replicated=["model"]
                )
                if self.rank == 0:
                    tqdm.write(f"💾 Snapshot saved at step {self.global_step}")
    
            if self.rank == 0:
                progress_bar.set_description(f"Epoch {epoch} | Batch {batch_idx}")
                progress_bar.set_postfix(loss=loss_value, avg=avg_running_loss)

            self.global_step += 1
            if self.logger:
                self.logger.log_metrics({
                    "train_loss": loss_value,
                    "avg_running_loss": avg_running_loss
                }, step=self.global_step)

        if self.rank == 0:
            progress_bar.close()
        
        if self.rank == 0:
            tqdm.write(f"✅ Epoch {epoch} | Final Avg Running Loss: {avg_running_loss:.4f}")
        
        return avg_running_loss

    def resume_latest_snapshot(self):
        if not os.path.exists(self.snapshot_dir):
            if self.rank == 0:
                print("📂 No snapshot directory found.")
            return False
        
        latest_path = os.path.join(self.snapshot_dir, "latest")
        if not os.path.exists(latest_path):
            if self.rank == 0:
                print("🚫 No snapshot found at 'latest'.")
            return False

        if self.rank == 0:
            print(f"📦 Resuming from snapshot at {latest_path}")

        # Restore model state on all ranks
        Snapshot(path=latest_path).restore(app_state=self.snapshot_app_state)
        self.model.train()
        return True


    def fit(self, num_epochs, resume_from_latest=True):
        if resume_from_latest:
            self.resume_latest_snapshot()

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            # Optional: save snapshot each epoch
            if dist.is_initialized():
                dist.barrier()
                Snapshot.take(
                    path=os.path.join(self.snapshot_dir, f"epoch_{epoch}"),
                    app_state=self.snapshot_app_state,
                    replicated=["model"]
                )

            if self.rank == 0:
                tqdm.write(f"📦 Epoch {epoch} snapshot saved")

        if self.rank == 0:
            tqdm.write(f"🏁 Finished training at epoch {num_epochs}")
        
        if self.logger:
            self.logger.log_metrics({
                "final_epoch": num_epochs,
                "final_loss": train_loss
            }, step=self.global_step)

        return train_loss
