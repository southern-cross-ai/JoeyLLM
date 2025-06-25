import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def main(rank, world_size):

        # ─── Per-process ENV ─────────────────────
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # ─── Select GPU ──────────────────────────
    torch.cuda.set_device(rank)

    # ─── Init DDP process group ──────────────
    dist.init_process_group(backend="nccl", init_method="env://")
    
    
    # try:
    #     if rank == 0:
    #         print("✅ Loaded Config:")

    #     wandbLogger.set_mode(cfg.wandb.mode)

    #     device = torch.device(f"cuda:{local_rank}")
        
    #     logger = Nonedd
    #     if rank == 0:
    #         logger = wandbLogger(
    #             project_name=cfg.wandb.project,
    #             config=OmegaConf.to_container(cfg, resolve=True)
    #         )
    #     if rank == 0:
    #         print("📦 Loading Dataset...")
        
    #     dataloader = get_dataloader(
    #         data_path=cfg.data.data_path,
    #         chunk_size=cfg.data.chunk_size,
    #         buffer_text_size=cfg.data.buffer_text_size,
    #         batch_size=cfg.data.batch_size,
    #         num_workers=cfg.data.num_workers,
    #         world_size=world_size,
    #         rank=rank
    #     )
    #     if rank == 0:
    #         print("🧠 Initializing Model...")
        
        
    #     model = JoeyLLM(
    #         vocab_size=cfg.model.vocab_size,
    #         max_seq_len=cfg.model.max_seq_len,
    #         embed_dim=cfg.model.embed_dim,
    #         num_layers=cfg.model.num_layers,
    #         num_heads=cfg.model.num_heads,
    #         dropout=cfg.model.dropout,
    #     ).to(device)

        
    #     if rank == 0:
    #         print("📈 Loading Optimizer")

    #     if world_size > 1:
    #         model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[local_rank] if torch.cuda.is_available() else None,
    #         )

    #     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    #     # scheduler = LossAdaptiveWarmupScheduler(
    #     #     optimizer,
    #     #     init_lr=2e-4,
    #     #     warmup_steps=1000,
    #     #     decay_factor=0.8,
    #     #     patience=5,
    #     #     window_size=1500
    #     # )

    #     scheduler = LossAdaptiveWarmupScheduler(
    #         optimizer=optimizer,
    #         init_lr=2e-4,
    #         warmup_steps=2000,        # ~1.3% of total
    #         decay_factor=0.8,
    #         patience=5,
    #         threshold=2e-4,
    #         window_size=1000,         # Better balance between sensitivity and smoothness
    #         min_lr=1e-6               # Optional but highly recommended
    #     )

    #     if logger:
    #         logger.watch_model(model, log="all", log_freq=10000)

    #     if rank == 0:
    #         print("🚀 Launching Trainer...")

    #     trainer = Trainer(
    #         model=model,
    #         dataloader=dataloader,
    #         optimizer=optimizer,
    #         logger=logger,
    #         scheduler=scheduler,
    #         device=device,
    #         rank=rank
    #     )

    #     trainer.fit(num_epochs=1, resume_from_latest=True)


    #     if rank == 0:
    #         print("🏁 Training complete!")

    #     if logger:
    #         logger.finish()

    # finally:
    #     if world_size > 1:
    #         cleanup_distributed()

    #     if rank == 0:
    dist.destroy_process_group()
    print("✅ Done")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)

    # Launch one process per GPU, each will run main(rank, world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)





