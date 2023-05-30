import argparse
import os
import yaml
from collections.abc import MutableMapping as Map


default_config = {
    "model": {
        "class": "AutoModelForCausalLM",
        "load": {
            "device_map": "auto"
        }
    },
    "tokenizer": {
        "class": "AutoTokenizer",
    },
}


def nested_update(d, v):
    for key in v:
        if key in d and isinstance(d[key], Map) and isinstance(v[key], Map):
            nested_update(d[key], v[key])
        else:
            d[key] = v[key]

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
    
    nested_update(config, default_config)

