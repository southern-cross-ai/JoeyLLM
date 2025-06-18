from torch.utils.data import Dataset
from torch.nn import Module, CrossEntropyLoss
import deepspeed

class Trainer():

    def __init__(
        self, 
        model: Module, 
        dataset: Dataset, 
        logger, 
        config_path : str ="utils/deepspeed_config.json"
    ):
        
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.loss_fn = CrossEntropyLoss()
        
        self.model_engine, self.optimizer, self.train_dataloader, self.scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=dataset,
            config=config_path
            )
        
        self.device = self.model_engine.device

    def epoch(self, epoch: int):
        self.model_engine.train()

        for step, batch in enumerate(self.train_dataloader):
            inputs = batch["inputs"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model_engine(inputs)
            loss = self.loss_fn(outputs, labels)

            self.model_engine.backward(loss)
            self.model_engine.step()
            
            # if self.model_engine.global_rank == 0:
            #     print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            self.epoch(epoch)