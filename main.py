import hydra
from omegaconf import DictConfig
from configs import JoeyConfig 

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Validate with Pydantic
    try:
        model_cfg = JoeyConfig(**cfg)
    except Exception as e:
        print("❌ Validation Error:")
        raise e

    print("✅ Pydantic Model Config:")
    print(model_cfg)

if __name__ == "__main__":
    main()

    
