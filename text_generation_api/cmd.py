import argparse
import os
import yaml

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
    
    assert "model" in config, "Model configuration not found"
    assert "name" in config["model"], "You need to at least specify the model name"

    if "name" not in config["tokenizer"]:
        config["tokenizer"]["name"] = config["model"]["name"]
    
    