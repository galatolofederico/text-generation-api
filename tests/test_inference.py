import unittest

from text_generation_api.utils import process_config
from text_generation_api.inference import Inference

test_config = "./example/opt-125.yaml"

class TestInference(unittest.TestCase):
    def test_inference(self):
        inference = Inference(process_config(test_config))
        result = inference.test()
        
        assert "generated" in result
        assert "stats" in result