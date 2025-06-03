import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
  '''
✅ Modular Trainer:
fit() – Runs full training + validation loop
_train_epoch() – One training pass (internal)
_validate_epoch() – One validation pass (internal)

Early stopping & checkpointing built-in
✅ Mixed precision (AMP) with torch.cuda.amp
✅ Logger support – easy to integrate with W&B, MLflow, etc.
✅ Flexible – swap datasets, models, optimizers with minimal changes!
'''
    def __init__(self, model, dataloader, val_dataloader, optimizer, scheduler=None, logger=None, device="cuda"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.scaler = GradScaler()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.dataloader):
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                msg = f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                print(msg)
                if self.logger:
                    self.logger.log_message(msg)
                    self.logger.log_metrics({"train_loss": loss.item()}, step=epoch * len(self.dataloader) + batch_idx)

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs = batch["inputs"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, labels)

                total_val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(self.val_dataloader)
        accuracy = 100.0 * correct / total

        msg = f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%"
        print(msg)
        if self.logger:
            self.logger.log_message(msg)
            self.logger.log_metrics({"val_loss": avg_val_loss, "val_accuracy": accuracy}, step=epoch)

        return avg_val_loss, accuracy

    def compute_loss(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)

    def save_checkpoint(self, path):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }
        if self.scheduler:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"Checkpoint loaded from {path}")

    def fit(self, num_epochs=20, early_stopping=3, checkpoint_path="checkpoint.pth"):
        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                self.save_checkpoint(checkpoint_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print("Early stopping triggered!")
                    break

            if self.scheduler:
                self.scheduler.step()

        print("Training complete!")
