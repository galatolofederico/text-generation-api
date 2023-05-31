import os
import yaml
from collections.abc import MutableMapping as Map

from text_generation_api.default import default_config

def process_config(config_file):
    assert os.path.exists(config_file), "Configuration file not found"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "backend" not in config:
        config["backend"] = "pytorch"
    
    assert config["backend"] in ["pytorch", "tensorflow"], "Backend not supported"
    if config["backend"] == "pytorch":
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed")
    elif config["backend"] == "tensorflow":
        try:
            import tensorflow
        except ImportError:
            raise ImportError("TensorFlow not installed")
    
    assert "name" in config["model"], "Model name not specified"
    nested_update(config, default_config)

    if not "name" in config["tokenizer"]:
        config["tokenizer"]["name"] = config["model"]["name"]

    return config

def nested_update(d, v):
    for key in v:
        if key in d and isinstance(d[key], Map) and isinstance(v[key], Map):
            nested_update(d[key], v[key])
        else:
            d[key] = v[key]