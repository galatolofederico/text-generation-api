import unittest
from fastapi.testclient import TestClient

from text_generation_api.inference import Inference
from text_generation_api.utils import process_config
from text_generation_api.server import create_app

test_config = "./example/opt-125.yaml"

class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)
        self.inference = {
            "test-endpoint": Inference(process_config(test_config))
        }
        self.app = TestClient(create_app(self.inference))

    def test_inference(self):
        pass