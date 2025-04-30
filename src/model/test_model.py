import torch
from model.joeyllm import JoeyLLM  # Adjust if your path is different

def Test_Model(model_configs, data_configs):
    # Instantiate the model
    model = JoeyLLM(model_configs)

    # Generate random input tensor
    input_ids = torch.randint(
        0,
        model_configs.vocab_size,
        (data_configs.batch_size, model_configs.max_seq_len)
    )

    # Forward pass
    logits = model(input_ids)

    # Output shapes
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")

    # Check output shape
    assert logits.shape == (data_configs.batch_size, model_configs.max_seq_len, model_configs.vocab_size), "Output shape is incorrect!"

