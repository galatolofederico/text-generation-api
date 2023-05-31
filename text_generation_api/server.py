from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union, Optional

def create_app(inferences, token=None):
    app = FastAPI()

    class GenerateRequest(BaseModel):
        prompt: Union[List[str], str]
        generate: Optional[dict] = None
        tokenize: Optional[dict] = None
        stop: Optional[dict] = None
    
    def verify_factory(_token):
        def verify_token(req: Request):
            if _token is None:
                return True
            
            token = req.headers["Authorization"]
            if token != _token:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized"
                )
        return verify_token
    
    @app.get("/hello")
    def hello():
        return "text-generation-api"

    @app.get("/generate/{model}")
    def generate(model, gen_args: GenerateRequest, authorized: bool = Depends(verify_factory(token))):
        return "ciao"

    return app