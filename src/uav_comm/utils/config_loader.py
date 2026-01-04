
import yaml
import os

def load_config(config_path="configs/env_config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
