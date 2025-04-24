import yaml
import sys
from pathlib import Path

def load_config(config_path: str = "config.yaml"):
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config file not found at: {config_path}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print("✅ Loaded configuration.")
    return config

def main():
    config = load_config()
    # print(config) 
    print("🧠 Starting JoeyLLM training process...")


    # TODO: Replace with actual model/tokenizer/training setup
    # print("Model type:", config["model"]["type"])

if __name__ == "__main__":
    main()

