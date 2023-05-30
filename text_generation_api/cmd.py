import argparse
import os
import yaml

from text_generation_api.utils import nested_update
from text_generation_api.inference import Inference

default_config = {
    "device": "cuda",
    "model": {
        "class": "AutoModelForCausalLM",
        "load": {
            "device_map": "auto"
        },
        "generate": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 512
        },
        "stop": {
            "ids": [],
            "words": []
        }
    },
    "tokenizer": {
        "class": "AutoTokenizer",
        "load": {
        },
        "tokenize": {
        },
    },
}

def main():
    parser = argparse.ArgumentParser(description='Text generation API server')

    parser.add_argument("config", help="Path to the configuration file", type=str)
    parser.add_argument("--host", help="Host to listen to", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="Port to listen to", type=int, default=8080)
    parser.add_argument("--token", help="Token to use for authentication", type=str, default=None)

    args = parser.parse_args()

    assert os.path.exists(args.config), "Configuration file not found"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    assert "name" in config["model"], "Model name not specified"
    nested_update(config, default_config)

    if not "name" in config["tokenizer"]:
        config["tokenizer"]["name"] = config["model"]["name"]

    print(config)
    inference = Inference(config)