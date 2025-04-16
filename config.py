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
        "tiny":    {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 512,  "num_heads": 8,  "num_layers": 2,  "dropout": 0.1},
        "small":   {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 768,  "num_heads": 12, "num_layers": 12, "dropout": 0.1},
        "medium":  {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1024, "num_heads": 16, "num_layers": 24, "dropout": 0.1},
        "large":   {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1280, "num_heads": 20, "num_layers": 36, "dropout": 0.1},
        "xl":      {"vocab_size": 50257, "max_seq_len": 1024, "hidden_dim": 1600, "num_heads": 25, "num_layers": 48, "dropout": 0.1}
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

