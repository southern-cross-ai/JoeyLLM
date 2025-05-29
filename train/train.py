import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb


class OneGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        *,
        device=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        epochs=10,
        checkpoint_path="./checkpoints",
        resume_from=None,
        save_every=1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.resume_from = resume_from
        self.save_every = save_every
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.epoch = 0
        self.step = 0

        self.load_checkpoint()

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }
        path = os.path.join(self.checkpoint_path, f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(checkpoint, path)
        print(f"✅ Saved checkpoint at epoch {self.epoch}")

    def load_checkpoint(self):
        path = self.resume_from or self.checkpoint_path
        if not os.path.exists(path):
            return

        try:
            if os.path.isdir(path):
                checkpoints = [f for f in os.listdir(path) if f.endswith(".pt")]
                if not checkpoints:
                    return
                latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                path = os.path.join(path, latest)

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.epoch = checkpoint["epoch"] + 1
            self.step = checkpoint["step"]

            print(f"🔄 Resumed training from epoch {self.epoch}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")


    def validate(self):
        if self.val_loader is None:
            return
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(
                self.val_loader,
                desc="🔍 Validation"
            )
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(self.device)
                outputs = self.model(input_ids[:, :-1])
                targets = input_ids[:, 1:]

                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )

                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(self.val_loader)
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        wandb.log({
            "val_loss": avg_val_loss,
            "epoch": self.epoch
        })


    def train(self):
        for epoch in range(self.epoch, self.epochs):
            self.model.train()
            epoch_loss = 0.0
            self.epoch = epoch

            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}"
            )
            self.optimizer.zero_grad()

            for i, batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                outputs = self.model(input_ids[:, :-1])
                targets = input_ids[:, 1:]

                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
                true_loss = loss.item()
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                epoch_loss += true_loss 
                self.step += 1

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix(
                    loss=true_loss 
                )

                wandb.log({
                    "loss": true_loss,
                    "epoch": epoch
                })

            print(f"📉 Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

            self.validate()


            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()
            
            self.scheduler.step()
                
