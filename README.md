# bitlinear
This project aims to provide a production-ready implementation of 1.58-bit layers for quantization-aware training and time-, memory-, and energy-efficient inference. It builds on the ideas from [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf).

# installation
Installation from PyPI:
```
pip install bitlinear
```

Installation from source:
```
git clone https://github.com/schneiderkamplab/bitlinear
cd bitlinear
pip install .
```

# usage
The usage is best explained by a short example:
```
from bitlinear import replace_modules
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/tiny-random-LlamaForCausalLM")
replace_modules(model)
```

More elaborate examples are available under `examples/classifier`, including training and evaluating a binary classifer:
```
pip install -r examples/classifier/requirements.txt
python examples/classifier/train.py
python examples/classifier/eval.py
```
There is also an MNIST classifier:
```
pip install -r examples/classifier/requirements.txt
python examples/mnist/train.py
```

# comparison to other work
There are other implementations of bit-linear layers, most of which get at least some of the details wrong at the time of this writing (April 2024).

The focus of this implementation is to develop:
* a flexible production-ready drop-in replacemenbt for torch.nn.LinearLayer,
* efficient fused kernels for training, and
* efficient fused kernels for inference with 2-bit weights and 8-bit activations.

Furthermore, this implementation is meant to serve as a testbed for research on low-bit quantization aware training and inference.

# future work
* further examples (vision, llm)
* efficient fused kernels for GPU/AVX/CPU training
* efficient fused kernels for GPU/AVX/CPU inferenc