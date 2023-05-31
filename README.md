# text-generation-api

> 📢 `text-generation-api` is a simple yet comprehensive REST API Server for text generation with huggingface models

**Couldn't be more easy to use 🔥**

**Comes with batteries included 🔋**

```python
from text_generation_api import Endpoint
tga = Endpoint("http://<host>:<port>")

result = tga.generate(
    model="gpt2",
    prompt="Here is a list of things I like to do:"
)
```

## Features 🏆

- Serve every [🤗 huggingface](https://huggingface.co/) model 🔥
- Batteries included🔋
- Nice one line serving and generation 😎

## Installation ⚙️

(optional) Create a `virtualenv`. You can use `conda` or whatever you like

```
virtualenv --python=python3.10 text-generation-api-env
. ./text-generation-api-env/bin/activate
```

Install `pytorch` (or `tensorflow`). Again, you can use whatever package manager you like

```
pip install "torch>=2.0.0"
```

Install `text-generation-api`

```
pip install text-generation-api
```

## Run the server 🌐

Create `yaml` config files for the models you want to serve

For example to serve `GPT2` the config file should look like

```yaml
model:
  name: gpt2
  class: GPT2Model

tokenizer:
  name: gpt2
  class: GPT2Tokenizer
```

To specify load arguments for the model or the tokenizer use the `load` key like:

```yaml
model:
  name: my-model
  class: GPT2Model
  load:
    device_map: auto
    trust_remote_code: True

tokenizer:
  name: gpt2
  class: GPT2Tokenizer
```

You can specify which device to use with `device` and which backend to use with `backend: pytorch` or `backend: tensorflow`

To run the inference server run

```
text-generation-api ./path/to/config1.yaml ./path/to/config2.yaml
```

For example:
```
text-generation-api ./example/gpt2.yaml  ./example/opt-125.yaml
```

## Run the client ✨

To run the inference on the remote server using the client simply:

```python
from text_generation_api import Endpoint
tga = Endpoint("http://<host>:<port>")

result = tga.generate(
    model="<MODEL-NAME>",
    prompt="<PROMPT>"
)
```

