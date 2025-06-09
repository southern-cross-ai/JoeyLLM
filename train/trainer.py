import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        scheduler=None,
        logger=None,
        device="cuda"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.scaler = GradScaler(device="cuda")

    def compute_loss(self, outputs, labels):
        """
        outputs: [B, T, vocab_size]
        labels:  [B, T]
        """
        criterion = torch.nn.CrossEntropyLoss()
        B, T, V = outputs.size()
        outputs = outputs.view(B * T, V)    # [B*T, V]
        labels = labels.view(B * T)         # [B*T]
        return criterion(outputs, labels)


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):

            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


            #pbar
            progress_bar.set_description(f"Epoch {epoch} | Batch {batch_idx}")
            progress_bar.set_postfix(loss=loss.item())
            
            # loger
            if self.logger:
                self.logger.log_message(msg)
                self.logger.log_metrics({
                    "train_loss": loss.item()
                }, step=epoch * len(self.dataloader) + batch_idx)
        
        # fix or iterable dataloader
        try:
            avg_loss = total_loss / len(self.dataloader)
        except TypeError:
            avg_loss = total_loss / (batch_idx + 1)
        
        tqdm.write(f"✅ Epoch {epoch} | Avg Training Loss: {avg_loss:.4f}")
        
        return avg_loss

    def save_checkpoint(self, path):
        checkpoint = {
            "model_state": self.model.state_dict(),
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

    def fit(self, num_epochs, checkpoint_path="checkpoints/checkpoint.pth"):
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.save_checkpoint(checkpoint_path)

            if self.scheduler:
                self.scheduler.step()
