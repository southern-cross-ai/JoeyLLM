import os
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from config import ModelConfig, TrainingConfig
from data import PGSDataset, get_dataloader
from model import GPT2Model

torch.cuda.empty_cache()

def initialize_training_components(model_cfg: ModelConfig, 
                                 train_cfg: TrainingConfig) -> tuple:
    """
    Initialize core training components with custom weights
    
    Args:
        model_cfg: Model architecture configuration
        train_cfg: Training hyperparameters
        
    Returns:
        tuple: (model, optimizer, scaler, dataloader, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize fresh model with custom weights
    model = GPT2Model(model_cfg).to(device)
    
    optimizer = AdamW(model.parameters(), 
                     lr=train_cfg.learning_rate,
                     weight_decay=train_cfg.weight_decay)
    
    scaler = torch.amp.GradScaler(enabled=train_cfg.mixed_precision)
    
    dataloader = get_dataloader(
        batch_size=train_cfg.batch_size,
        sample_fraction=train_cfg.sample_fraction
    )
    
    return model, optimizer, scaler, dataloader, device

def train_epoch(model: torch.nn.Module, 
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler,
               device: torch.device,
               train_cfg: TrainingConfig) -> float:
    """
    Execute one training epoch with custom initialized model
    (Implementation remains same as previous)
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch['input_ids'].to(device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', 
                              dtype=torch.float16, 
                              enabled=train_cfg.mixed_precision):
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                inputs.view(-1)
            )
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if batch_idx % train_cfg.log_interval == 0:
            wandb.log({"batch_loss": loss.item()})
    
    return total_loss / len(dataloader)

def main():
    """
    Main training execution with pure custom initialization
    """
    model_cfg = ModelConfig(model_size="small")
    train_cfg = TrainingConfig(
        batch_size=8,
        epochs=1000,
        learning_rate=3e-4,
        save_dir="./checkpoints",
        sample_fraction=0.25,
        mixed_precision=True
    )
    
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    
    wandb.init(project="gpt2-custom-init", config={
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__
    })
    
    model, optimizer, scaler, dataloader, device = initialize_training_components(
        model_cfg, train_cfg
    )
    
    best_loss = float('inf')
    for epoch in range(train_cfg.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, 
                             scaler, device, train_cfg)
        
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(train_cfg.save_dir, 
                      f"best_model_epoch_{epoch}.pt"))
            
if __name__ == "__main__":
    main()