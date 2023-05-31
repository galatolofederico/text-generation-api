import argparse
import os
import yaml

from text_generation_api.default import default_config
from text_generation_api.utils import nested_update
from text_generation_api.inference import Inference


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

def main():
    parser = argparse.ArgumentParser(description='Text generation API server')

    parser.add_argument("configs", nargs="+", help="Path to the configuration files", type=str)
    parser.add_argument("--host", help="Host to listen to", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="Port to listen to", type=int, default=8080)
    parser.add_argument("--token", help="Token to use for authentication", type=str, default=None)
    parser.add_argument("--test", help="Run in test mode", action="store_true")

    args = parser.parse_args()

    inferences = []
    for config_file in args.configs:
        inferences.append(Inference(process_config(config_file)))

    if args.test:
        for inference in inferences:
            print("====")
            print("Testing :"+inference.config["model"]["name"])
            print("== CONFIG ==")
            print(inference.config)
            print("== INFERENCE ==")
            print(inference.test())
            print("====")