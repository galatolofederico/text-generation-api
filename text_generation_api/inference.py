import transformers
import time

from text_generation_api.utils import nested_update

class GeneratorStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, *, prompt_tokens, ids=[], words=[], tokenizer=None):
        self.prompt_tokens = prompt_tokens
        self.ids = ids
        self.words = words
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores):
        if self.ids is not None:
            do_stop = [False for _ in range(len(input_ids))]
            for i, (t, p) in enumerate(zip(input_ids, self.prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in self.ids:
                    if stop_id in g:
                        do_stop[i] = True

            if all(do_stop):
                return True

        if self.words is not None:
            do_stop = [False for _ in range(len(input_ids))]
            for i, (t, p) in enumerate(zip(input_ids, self.prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in self.words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False


class Inference:
    def __init__(self, config, debug=False):
        self.config = config.copy()
        self.debug = debug

        self.device = self.config["device"]
        if config["backend"] == "pytorch":
            self.backend = "pt"
        elif config["backend"] == "tensorflow":
            self.backend = "tf"
        else:
            raise ValueError("Backend not supported")

        tokenizer_name = self.config["tokenizer"]["name"]
        model_name = self.config["model"]["name"]

        if self.debug:
            print("Tokenizer name:", tokenizer_name)
            print("Tokenizer loading class: ", self.config["tokenizer"]["class"])
            print("Tokenizer loading arguments: ", self.config["tokenizer"]["load"])

        tokenizer_cls = getattr(transformers, self.config["tokenizer"]["class"])
        self.tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_name,
            **self.config["tokenizer"]["load"]
        )

        if self.debug:
            print("Model name:", model_name)
            print("Model loading class: ", self.config["model"]["class"])
            print("Model loading arguments: ", self.config["model"]["load"])

        model_cls = getattr(transformers, self.config["model"]["class"])
        self.model = model_cls.from_pretrained(
            model_name,
            **self.config["model"]["load"]
        ).to(self.device)
        
        if "peft" in self.config:
            from peft import PeftModel
            peft_model_name = self.config["peft"]["name"]

            if self.debug:
                print("PEFT model name:", peft_model_name)
                print("PEFT loading arguments: ", self.config["peft"]["load"])
                
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_model_name,
                **self.config["peft"]["load"]
            ).to(self.device)

    def test(self):
        return self.generate({"prompt": self.config["test"]})

    def generate(self, args):
        assert "prompt" in args, "prompt is required"
        if isinstance(args["prompt"], str):
            args["prompt"] = [args["prompt"]]

        if "generate" not in args:
            args["generate"] = dict()
        if "tokenize" not in args:
            args["tokenize"] = dict()
        if "stop" not in args:
            args["stop"] = dict()

        if args["generate"] is None:
            args["generate"] = dict()
        if args["tokenize"] is None:
            args["tokenize"] = dict()
        if args["stop"] is None:
            args["stop"] = dict()

        nested_update(args["generate"], self.config["model"]["generate"])
        nested_update(args["tokenize"], self.config["tokenizer"]["tokenize"])
        nested_update(args["stop"], self.config["model"]["stop"])

        if self.debug:
            print("Prompt:", args["prompt"])
            print("Generate arguments:", args["generate"])
            print("Tokenize arguments:", args["tokenize"])
            print("Stop arguments:", args["stop"])

        t0 = time.time()
        inputs = self.tokenizer(
            args["prompt"],
            return_tensors=self.backend,
            **args["tokenize"]
        ).to(self.device)
        tokens = self.model.generate(
            **inputs,
            **args["generate"],
            stopping_criteria=transformers.StoppingCriteriaList([
                GeneratorStoppingCriteria(
                    prompt_tokens=inputs["input_ids"],
                    tokenizer=self.tokenizer,
                    **args["stop"]
                ),
            ])
        )
        generated = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        t1 = time.time()

        stats = {
            "total_seconds": t1 - t0,
            "tok/s": max([len(t) for t in tokens]) / (t1 - t0)
        }

        return {"generated": generated, "stats": stats}
