from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union

def create_app(inferences, token=None):
    app = FastAPI()

    class GenerateRequest(BaseModel):
        prompt: Union[List[str], str]
        generate: dict = None
        tokenize: dict = None
        stop: dict = None
    
    def verify_token(req):
        if token == "":
            return True
        
        token = req.headers["Authorization"]
        if token != token:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
    
    @app.get("/hello")
    def hello():
        return "text-generation-api"

    @app.get("/generate/{model}")
    def generate(gen_args: GenerateRequest, authorized: bool = Depends(verify_token)):
        print(gen_args)
        return "ciao"

    return app