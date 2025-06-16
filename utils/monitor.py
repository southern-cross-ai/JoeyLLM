import os
from tqdm import tqdm
import deepspeed
import torch
from torch import nn
from torch.utils.data import DataLoader


class TrainerMonitor:
    def __init__(self, rank, log_interval=100):
        self.rank = rank
        self.global_step = 0
        self.log_interval = log_interval
        self.progress = None

    def init_epoch(self, epoch, dataloader):
        if self.rank == 0:
            self.progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        else:
            self.progress = dataloader
        return self.progress

    def update(self, epoch, batch_idx, loss, avg_loss):
        if self.rank == 0:
            self.progress.set_description(f"Epoch {epoch} | Batch {batch_idx}")
            self.progress.set_postfix(loss=loss, avg=avg_loss)

    def log_metrics(self, metrics: dict):
        if self.rank == 0:
            print(f"📊 Step {self.global_step}: {metrics}")

    def close_epoch(self, epoch, avg_loss):
        if self.rank == 0:
            self.progress.close()
            tqdm.write(f"✅ Epoch {epoch} | Final Avg Running Loss: {avg_loss:.4f}")

    def final_summary(self, epoch, loss):
        if self.rank == 0:
            tqdm.write(f"🌟 Finished training at epoch {epoch}")
            print(f"Final Loss: {loss:.4f}")
