import argparse

from text_generation_api.utils import nested_update, process_config
from text_generation_api.inference import Inference
from text_generation_api.server import create_app

OKGREEN = '\033[92m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def main():
    parser = argparse.ArgumentParser(description='Text generation API server')

    parser.add_argument("configs", nargs="+", help="Path to the configuration files", type=str)
    parser.add_argument("--host", help="Host to listen to", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="Port to listen to", type=int, default=3000)
    parser.add_argument("--token", help="Token to use for authentication", type=str, default=None)
    parser.add_argument("--test", help="Run in test mode", action="store_true")
    parser.add_argument("--debug", help="Run in test mode", action="store_true")

    args = parser.parse_args()

    inferences = dict()
    for config_file in args.configs:
        print("Loading "+config_file+"...")
        inference = Inference(process_config(config_file), debug=args.debug)
        if "endpoint" not in inference.config:
            endpoint = inference.config["model"]["name"].replace("/", "-")
        else:
            endpoint = inference.config["endpoint"]
        inferences[endpoint] = inference
    
    if args.test:
        for endpoint, inference in inferences.items():
            print("====")
            print("Testing :"+inference.config["model"]["name"])
            print("== CONFIG ==")
            print(inference.config)
            print("== INFERENCE ==")
            print(inference.test())
            print("====")
    else:
        import uvicorn

        print("Starting server...")
        for endpoint, inference in inferences.items():
            print(BOLD+OKGREEN+"Model "+inference.config["model"]["name"]+" available at http://"+args.host+":"+str(args.port)+"/generate/"+endpoint+ENDC)
        app = create_app(inferences, token=args.token)
        uvicorn.run(app, host=args.host, port=args.port)