from dataclasses import dataclass

class ModelConfig:
    """
    GPT-2 model architecture configuration with preset sizes
    
    Attributes:
        model_configs (dict): Preset configurations for different model sizes
        vocab_size (int): Size of token vocabulary (default: 50257)
        max_seq_len (int): Maximum sequence length (default: 1024)
        hidden_dim (int): Dimension of hidden layers (default: 768)
        num_heads (int): Number of attention heads (default: 12)
        num_layers (int): Number of transformer blocks (default: 12)
        dropout (float): Dropout probability (default: 0.1)
        
    Methods:
        __init__: Initialize configuration from preset or custom parameters
    """
    
    model_configs = {
        "tiny":    {"vocab_size": 100256, "max_seq_len": 2048, "hidden_dim": 512,  "num_heads": 8,  "num_layers": 2,  "dropout": 0.1},
        "small":   {"vocab_size": 100256, "max_seq_len": 2048, "hidden_dim": 768,  "num_heads": 12, "num_layers": 12, "dropout": 0.1},
        "medium":  {"vocab_size": 100256, "max_seq_len": 2048, "hidden_dim": 1024, "num_heads": 16, "num_layers": 24, "dropout": 0.1},
        "large":   {"vocab_size": 100256, "max_seq_len": 2048, "hidden_dim": 1280, "num_heads": 20, "num_layers": 36, "dropout": 0.1},
        "xl":      {"vocab_size": 100256, "max_seq_len": 2048, "hidden_dim": 1600, "num_heads": 25, "num_layers": 48, "dropout": 0.1}
    }

    def __init__(self, model_size: str = None, **kwargs):
        """
        Initialize model configuration
        
        Args:
            model_size (str): Preset size name (tiny/small/medium/large/xl)
            **kwargs: Custom configuration parameters to override defaults
        """
        if model_size:
            assert model_size in self.model_configs, f"Invalid model size: {model_size}"
            self.__dict__.update(self.model_configs[model_size])
        else:
            defaults = {
                "vocab_size": 50257,
                "max_seq_len": 1024,
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "dropout": 0.1
            }
            defaults.update(kwargs)
            self.__dict__.update(defaults)

@dataclass
class TrainingConfig:
    """
    Training process configuration
    
    Attributes:
        batch_size (int): Samples per batch (default: 8)
        epochs (int): Total training epochs (default: 1000)
        learning_rate (float): Initial learning rate (default: 3e-4)
        weight_decay (float): L2 regularization (default: 0.01)
        warmup_steps (int): Linear warmup steps (default: 1000)
        log_interval (int): Loss logging frequency (default: 100)
        save_dir (str): Model checkpoint directory (default: "./checkpoints")
        sample_fraction (float): Dataset sampling ratio (default: 0.25)
        use_fsdp (bool): Enable Fully Sharded Data Parallel (default: False)
        use_ddp (bool): Enable Distributed Data Parallel (default: False)
        mixed_precision (bool): Use FP16 training (default: True)
        gradient_accumulation_steps (int): Accumulate gradients over steps (default: 1)
        checkpoint_interval (int): Save interval in epochs (default: 1)
        max_grad_norm (float): Gradient clipping threshold (default: 1.0)
    """
    batch_size: int = 8
    epochs: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    log_interval: int = 10
    save_dir: str = "./checkpoints"
    sample_fraction: float = 0.25
    use_fsdp: bool = False
    use_ddp: bool = False
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 1
    max_grad_norm: float = 1.0

@dataclass
class GenerationConfig:
    """
    Text generation configuration
    
    Attributes:
        max_length (int): Maximum tokens to generate (default: 100)
        temperature (float): Sampling temperature (default: 0.9)
        top_p (float): Nucleus sampling threshold (default: 0.92)
        max_input_tokens (int): Maximum allowed input tokens (default: 512)
        do_sample (bool): Enable sampling (default: True)
        num_beams (int): Beam search width (default: 1)
    """
    max_length: int = 100
    temperature: float = 0.9
    top_p: float = 0.92
    max_input_tokens: int = 512
    do_sample: bool = True
    num_beams: int = 1