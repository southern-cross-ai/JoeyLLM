import 
from torch.utils.data import Dataset
from torch.nn import Module

class Trainer():

    def __init__(
        self, 
        model: Module, 
        dataset: Dataset, 
        logger, 
        config_path : str ="utils/deepspeed_config.json"
    ):
        
        self.model = model,
        self.dataset = dataset,
        self.logger = logger


        
        
    model_engine, optimizer, train_loader, scheduler = trainer.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config="ds_config.json"  # just pass path directly
    )

    for epoch in range(2):
        for step, (x, y) in enumerate(train_loader):
            x = x.to(model_engine.device)
            y = y.to(model_engine.device)

            outputs = model_engine(x)
            loss = loss_fn(outputs, y)

            model_engine.backward(loss)
            model_engine.step()

            if model_engine.global_rank == 0:
                print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()