import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import re


class Trainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        logger,
        scheduler=None,
        device="cuda"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.device = device
        self.scaler = GradScaler(device=self.device)
        self.loss_milestones = [6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        self.next_milestone_idx = 0
        self.global_step = 0


    def compute_loss(self, outputs, labels):
        """
        outputs: [B, T, vocab_size]
        labels:  [B, T]
        """
        B, T, V = outputs.size()
        outputs = outputs.view(B * T, V)    # [B*T, V]
        labels = labels.view(B * T)         # [B*T]
        return self.criterion(outputs, labels)

    def _train_epoch(self, epoch):
        self.model.train()
        self.running_loss = 0.0
        self.total_batches = 0

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)

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

            # === 🔁 Milestone Logic ===
            loss_value = loss.item()
            self.running_loss += loss_value
            self.total_batches += 1
            avg_running_loss = self.running_loss / self.total_batches

            # 🧠 Check against current milestone
            if self.next_milestone_idx < len(self.loss_milestones):
                milestone = self.loss_milestones[self.next_milestone_idx]
                if avg_running_loss < milestone:
                    save_path = f"checkpoints/below_{milestone:.1f}_loss.pth"
                    self.save_checkpoint(save_path)
                    tqdm.write(f"📉 Saved checkpoint at avg loss < {milestone:.1f} (avg: {avg_running_loss:.4f})")
                    self.next_milestone_idx += 1

            # === Progress + Logging ===
            progress_bar.set_description(f"Epoch {epoch} | Batch {batch_idx}")
            progress_bar.set_postfix(loss=loss_value, avg=avg_running_loss)
            
            # logger
            self.global_step += 1
            if self.logger:
                self.logger.log_metrics({
                    "train_loss": loss_value,
                    "avg_running_loss": avg_running_loss
                }, step=self.global_step)

        progress_bar.close()
        tqdm.write(f"✅ Epoch {epoch} | Final Avg Running Loss: {avg_running_loss:.4f}")
        return avg_running_loss


    def save_checkpoint(self, path):
        print(f"📝 Attempting to save checkpoint to: {os.path.abspath(path)}")

        model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        checkpoint = {
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict()
        }
        if self.scheduler:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"✅ Checkpoint loaded from {path}")

    def resume_from_best_checkpoint(self, folder="checkpoints"):
        """
        Finds and loads the checkpoint with the lowest milestone loss.
        """
        if not os.path.exists(folder):
            print(f"📂 No checkpoint folder found at {folder}")
            return False

        pattern = re.compile(r"below_(\d+\.\d+)_loss\.pth")
        best_loss = float("inf")
        best_path = None

        for filename in os.listdir(folder):
            match = pattern.match(filename)
            if match:
                loss = float(match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_path = os.path.join(folder, filename)

        if best_path:
            print(f"📦 Resuming from best checkpoint: {best_path}")
            self.load_checkpoint(best_path)
            
            # 🧠 Update milestone index to skip saved ones
            for i, milestone in enumerate(self.loss_milestones):
                if milestone <= best_loss:
                    self.next_milestone_idx = i + 1
                    break
            else:
                self.next_milestone_idx = len(self.loss_milestones)


            # Add this:
            print(f"⏭️ Skipping to milestone index: {self.next_milestone_idx} ({self.loss_milestones[self.next_milestone_idx] if self.next_milestone_idx < len(self.loss_milestones) else 'done'})")

            return True
            
        else:
            print("🚫 No matching loss-based checkpoints found.")
            return False

    def fit(self, num_epochs, checkpoint_path="checkpoints/checkpoint.pth", resume_from_best=True):
        if resume_from_best:
            self.resume_from_best_checkpoint(folder=os.path.dirname(checkpoint_path))

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint after each epoch
            self.save_checkpoint(checkpoint_path)

        # Log completion of final epoch
        tqdm.write(f"\ud83c\udfce\ufe0f Finished training at epoch {num_epochs}")
        if self.logger:
            self.logger.log_metrics({
                "final_epoch": num_epochs,
                "final_loss": train_loss
            }, step=self.global_step)

        return train_loss