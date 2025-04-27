import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import Dataloaders


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("----------Configs!----------")
    print(OmegaConf.to_yaml(cfg))

    train_loader, val_loader, _ = Dataloaders(cfg)

    print("----------Loading Model to GPU!----------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = JoeyLLM(
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout
    )

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 1  # adjust as needed

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("----------Training Done!----------")

if __name__ == "__main__":
    main()

