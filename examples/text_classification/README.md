# BitMLP for text classification

We start with a wide two-layer MLP for text classification, as proposed in [Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP](https://aclanthology.org/2022.acl-long.279) (Galke & Scherp, ACL 2022). We replace the `torch.nn.Linear` output layer by `BitLinear` and experiment with 1.58b-mean 1.58b-median quantization (while the initial input-to-hidden layer is implemented as an embedding layer).

##  Usage
1. Fetch the data folder from https://github.com/lgalke/text_gcn
2. Make sure the following packages are installed `torch`, `transformers`, `numpy`, `sklearn`
2. `bash run.bash`

## Results

| **Model** | **20ng** | **R8** | **R52** | **ohsumed** | **MR** | **avg.** |
|:--|:--|:--|:--|:--|:--|:--|
| TF-IDF WideMLP | 84.20 | 97.08 | 93.67 | 66.06 | 76.32 |83.47 |
| WideMLP | 83.31 | 97.27 | 93.89 | 63.95 | 76.72 | 83.03 |
| WideMLP-1.58b-mean | 79.89 | 97.40 | 93.54 | 60.75 | 77.10 |  81.74
| WideMLP-1.58b-median | 80.08 | 97.35 | 94.20 | 62.28 | 76.14 | 82.01 |

WideMLP-1.58-mean achieves 98.4% of WideMLP performance.

WideMLP-1.58-median achieves 98.8% of WideMLP performance.
