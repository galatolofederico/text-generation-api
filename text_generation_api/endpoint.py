import requests

class Endpoint:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        res = requests.get(self.endpoint+"/hello")
        assert res.json() == "text-generation-api"

    def generate(self, *, model, prompt, generate=None, tokenize=None, stop=None):
        res = requests.get(
            self.endpoint+"/generate/"+model,
            json={
                "prompt": prompt,
                "generate": generate,
                "tokenize": tokenize,
                "stop": stop,
            }
        )
        print(res.json())