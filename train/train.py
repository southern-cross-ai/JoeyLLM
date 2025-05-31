import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
import wandb

class OneGPUTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader=None, *, device=None, learning_rate=1e-4, weight_decay=0.01, gradient_accumulation_steps=1, gradient_clip_norm=1.0, epochs=10, warmup_steps=500, save_every=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.save_every = save_every
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = GradScaler()

        # Warmup scheduler: linear warmup for 'warmup_steps' steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss()

        self.epoch = 0
        self.step = 0

        print(f"Using device: {self.device}")

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}")
        self.optimizer.zero_grad()

        for i, batch in pbar:
            input_ids = batch["input_ids"].to(self.device)

            with autocast():
                outputs = self.model(input_ids[:, :-1])
                targets = input_ids[:, 1:]
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
                true_loss = loss.item()
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            epoch_loss += true_loss
            self.step += 1

            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            pbar.set_postfix(loss=true_loss)
            wandb.log({"loss": true_loss, "epoch": epoch})

        print(f"📉 Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}")

    def validate(self, epoch):
        if self.val_loader is None:
            return None
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="🔍 Validation"):
                input_ids = batch["input_ids"].to(self.device)
                outputs = self.model(input_ids[:, :-1])
                targets = input_ids[:, 1:]
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
        return avg_val_loss

    def fit(self):
        for epoch in range(self.epoch, self.epochs):
            self.train(epoch)
            val_loss = self.validate(epoch)

            if (epoch + 1) % self.save_every == 0:
                print(f"✅ Model checkpoint for epoch {epoch + 1} (no actual checkpointing code)")

            self.epoch = epoch + 1
