from torchsnapshot import Stateful
from transformers import get_cosine_schedule_with_warmup
from collections import deque

class StreamingCosineWithLossFallback(Stateful):
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        factor=0.2,
        patience=3,
        threshold=1e-4,
        window_size=500
    ):
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.window_size = window_size

        self.recent_losses = deque(maxlen=window_size)
        self.prev_avg_loss = None
        self.bad_window_count = 0
        self.current_step = 0

    def step(self, loss=None):
        self.scheduler.step()
        self.current_step += 1

        # Update recent losses and check improvement
        if loss is not None:
            self.recent_losses.append(loss)

            if len(self.recent_losses) == self.window_size:
                avg_loss = sum(self.recent_losses) / self.window_size

                if self.prev_avg_loss is not None:
                    improvement = self.prev_avg_loss - avg_loss
                    if improvement < self.threshold:
                        self.bad_window_count += 1
                    else:
                        self.bad_window_count = 0  # reset if improved

                self.prev_avg_loss = avg_loss

                if self.bad_window_count >= self.patience:
                    print(f"🔁 Plateau detected — reducing LR by factor {self.factor}")
                    for group in self.optimizer.param_groups:
                        group['lr'] *= self.factor
                    self.bad_window_count = 0  # reset

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "prev_avg_loss": self.prev_avg_loss,
            "bad_window_count": self.bad_window_count,
            "recent_losses": list(self.recent_losses),
        }

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state["scheduler"])
        self.current_step = state["current_step"]
        self.prev_avg_loss = state["prev_avg_loss"]
        self.bad_window_count = state["bad_window_count"]
        self.recent_losses.clear()
        self.recent_losses.extend(state["recent_losses"])

from collections import deque
import torch
from torchsnapshot import Stateful

class LossAdaptiveWarmupScheduler(Stateful):
    def __init__(
        self,
        optimizer,
        init_lr,
        warmup_steps=1000,
        patience=2,
        threshold=1e-4,
        cooldown=0,
        decay_factor=0.7,
        warmup_multiplier=1.2,
        window_size=500
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.decay_factor = decay_factor
        self.warmup_multiplier = warmup_multiplier
        self.window_size = window_size

        self.loss_window = deque(maxlen=window_size)
        self.global_step = 0
        self.stage = "warmup"
        self.current_lr = init_lr
        self.lr_target = init_lr
        self.warmup_start_lr = 0.0
        self.warmup_step = 0
        self.bad_epoch_count = 0
        self.cooldown_counter = 0

        self._set_lr(self.current_lr)

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_lr = lr

    def step(self, loss):
        self.global_step += 1
        self.loss_window.append(loss)

        if self.stage == "warmup":
            self.warmup_step += 1
            progress = self.warmup_step / self.warmup_steps
            new_lr = self.warmup_start_lr + progress * (self.lr_target - self.warmup_start_lr)
            self._set_lr(new_lr)

            if self.warmup_step >= self.warmup_steps:
                self.stage = "monitor"
                self.prev_avg_loss = sum(self.loss_window) / len(self.loss_window)
        elif self.stage == "monitor":
            if len(self.loss_window) == self.window_size:
                avg_loss = sum(self.loss_window) / self.window_size
                improvement = self.prev_avg_loss - avg_loss
                if improvement < self.threshold:
                    self.bad_epoch_count += 1
                else:
                    self.bad_epoch_count = 0
                self.prev_avg_loss = avg_loss

                if self.bad_epoch_count >= self.patience and self.cooldown_counter == 0:
                    # Decay LR and restart warmup
                    new_base_lr = self.current_lr * self.decay_factor
                    new_target = new_base_lr * self.warmup_multiplier
                    print(f"🔁 Plateau detected — decaying LR to {new_base_lr:.6f}, restarting warmup to {new_target:.6f}")
                    self.stage = "warmup"
                    self.warmup_step = 0
                    self.lr_target = new_target
                    self.warmup_start_lr = new_base_lr
                    self._set_lr(new_base_lr)
                    self.bad_epoch_count = 0
                    self.cooldown_counter = self.cooldown
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

    def get_last_lr(self):
        return [self.current_lr]

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "stage": self.stage,
            "current_lr": self.current_lr,
            "lr_target": self.lr_target,
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_step": self.warmup_step,
            "bad_epoch_count": self.bad_epoch_count,
            "cooldown_counter": self.cooldown_counter,
            "prev_avg_loss": getattr(self, "prev_avg_loss", None),
            "loss_window": list(self.loss_window),
        }

    def load_state_dict(self, state):
        self.global_step = state["global_step"]
        self.stage = state["stage"]
        self.current_lr = state["current_lr"]
        self.lr_target = state["lr_target"]
        self.warmup_start_lr = state["warmup_start_lr"]
        self.warmup_step = state["warmup_step"]
        self.bad_epoch_count = state["bad_epoch_count"]
        self.cooldown_counter = state["cooldown_counter"]
        self.prev_avg_loss = state["prev_avg_loss"]
        self.loss_window = deque(state["loss_window"], maxlen=self.window_size)
        self._set_lr(self.current_lr)
