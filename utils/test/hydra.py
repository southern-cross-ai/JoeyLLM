import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path

# Set working directory to project root
ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

print("Working directory manually set to:", os.getcwd())

@hydra.main(
    config_path=str(ROOT_DIR / "src" / "configs"),
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("✅ Loading Configs:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()

