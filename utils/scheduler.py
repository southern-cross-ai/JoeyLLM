from torchsnapshot import Stateful
from transformers import get_cosine_schedule_with_warmup
from collections import deque

class StreamingCosineWithLossFallback(Stateful):
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        factor=0.5,
        patience=3,
        threshold=1e-4,
        window_size=1000
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

