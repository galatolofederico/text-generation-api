default_config = {
    "device": "cuda",
    "backend": "pytorch",
    "test": "text-generation-api is a",
    "model": {
        "class": "AutoModelForCausalLM",
        "load": {
        },
        "generate": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 512
        },
        "stop": {
            "ids": None,
            "words": None
        }
    },
    "tokenizer": {
        "class": "AutoTokenizer",
        "load": {
        },
        "tokenize": {
        },
    },
}