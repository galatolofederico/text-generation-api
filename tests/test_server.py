import unittest
import uvicorn
import contextlib
import threading
import time

from text_generation_api import Endpoint
from text_generation_api.inference import Inference
from text_generation_api.utils import process_config
from text_generation_api.server import create_app

test_host = "127.0.0.1"
test_port = 8999
test_config = "./example/opt-125.yaml"


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

class TestInference(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInference, self).__init__(*args, **kwargs)
        self.inference = {
            "test-endpoint": Inference(process_config(test_config), debug=True)
        }
        app = create_app(self.inference)
        config = uvicorn.Config(app, host=test_host, port=test_port)
        self.server = Server(config=config)

    def test_inference(self):
        with self.server.run_in_thread():
            endpoint = Endpoint("http://"+test_host+":"+str(test_port))
            result = endpoint.generate(
                model="test-endpoint",
                prompt="Here is a list of things I like to do:"
            )
            assert "generated" in result
            assert "stats" in result