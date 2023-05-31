from fastapi import FastAPI

def create_app(inferences, token=None):
    app = FastAPI()
    
    @app.get("/generate/{model}")
    def generate(model):
        return model

    return app