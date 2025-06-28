import torch
from torch.amp import autocast, GradScaler
from torchsnapshot import Snapshot, Stateful
import torch.distributed as dist
from tqdm import tqdm
import shutil
import os
import glob

class Trainer(Stateful):
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
        self.save_interval = 5000  
        self.retention_limit = 2
        self.pending_snapshot = None

        self.snapshot_dir = "snapshots"
        self.snapshot_app_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "trainer": self
        }
        if self.scheduler:
            self.snapshot_app_state["scheduler"] = self.scheduler

    def state_dict(self):
        return {"global_step": self.global_step}

    def load_state_dict(self, state):
        self.global_step = state.get("global_step", 0)

    def compute_loss(self, outputs, labels):
        B, T, V = outputs.size()
        outputs = outputs.view(B * T, V)
        labels = labels.view(B * T)
        return self.criterion(outputs, labels)

    def _rotate_snapshots(self):
        if self.rank != 0:
            return

        all_step_dirs = sorted(
            glob.glob(os.path.join(self.snapshot_dir, "step_*")),
            key=os.path.getmtime,
            reverse=True
        )

        for old_dir in all_step_dirs[self.retention_limit:]:
            tqdm.write(f"🧹 Removing old snapshot: {old_dir}")
            shutil.rmtree(old_dir, ignore_errors=True)

    def _save_snapshot(self, step: int = None):
        snapshot_path_latest = os.path.join(self.snapshot_dir, "latest")
        snapshot_path_rotated = os.path.join(self.snapshot_dir, f"step_{step}") if step is not None else None

        if self.rank == 0 and os.path.exists(snapshot_path_latest):
            shutil.rmtree(snapshot_path_latest)

        if dist.is_initialized():
            dist.barrier()

        if self.pending_snapshot and not self.pending_snapshot.done():
            if self.rank == 0:
                tqdm.write("⏳ Waiting for previous async snapshot to finish...")
            self.pending_snapshot.wait()

        self.pending_snapshot = Snapshot.async_take(
            path=snapshot_path_latest,
            app_state=self.snapshot_app_state,
            replicated=["model"]
        )

        if snapshot_path_rotated:
            Snapshot.take(
                path=snapshot_path_rotated,
                app_state=self.snapshot_app_state,
                replicated=["model"]
            )

        if self.rank == 0:
            tqdm.write(f"📂 Async snapshot started at step {step if step else '[unknown]'}")
            self._rotate_snapshots()

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
            if self.scheduler:
                self.scheduler.step(loss=loss_value)
            self.running_loss += loss_value
            self.total_batches += 1
            avg_running_loss = self.running_loss / self.total_batches

            if self.logger and self.global_step % 100 == 0:
                self.logger.log_metrics({
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                }, step=self.global_step)

            if self.global_step % self.save_interval == 0:
                self._save_snapshot(step=self.global_step)

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
            print(f"📆 Resuming from snapshot at {latest_path}")

        Snapshot(path=latest_path).restore(app_state=self.snapshot_app_state)
        self.model.train()
        return True

    def fit(self, num_epochs, resume_from_latest=True):
        if resume_from_latest:
            self.resume_latest_snapshot()

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)

            if self.rank == 0:
                # 💾 Save model checkpoint at end of epoch
                model_path = os.path.join(self.snapshot_dir, f"epoch_{epoch}_model.pt")
                torch.save(self.model.state_dict(), model_path)
                tqdm.write(f"💾 Saved model checkpoint at {model_path}")
                tqdm.write(f"📆 Epoch {epoch} completed")

        if self.rank == 0:
            tqdm.write(f"🌟 Finished training at epoch {num_epochs}")

        if self.pending_snapshot and not self.pending_snapshot.done():
            if self.rank == 0:
                tqdm.write("🕓 Waiting for final async snapshot to complete...")
            self.pending_snapshot.wait()

        if self.logger:
            self.logger.log_metrics({
                "final_epoch": num_epochs,
                "final_loss": train_loss
            }, step=self.global_step)

        return train_loss

