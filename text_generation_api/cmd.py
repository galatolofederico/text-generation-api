import argparse

def main():
    parser = argparse.ArgumentParser(description='Text generation API server')

    parser.add_argument("config", help="Path to the configuration file", type=str)
    parser.add_argument("--host", help="Host to listen to", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="Port to listen to", type=int, default=8080)
    parser.add_argument("--token", help="Token to use for authentication", type=str, default=None)

    args = parser.parse_args()

    print(args)