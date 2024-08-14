# Node Classification Example

In node classification, the model should learn a mapping from node features $X$ and node connectivity $A$ to derive a class labels for each node $y$.  We experiment with a simple yet powerful model for node classification: Simplified Graph Convolution (SGC) by [Wu et al., ICML 2019](http://proceedings.mlr.press/v97/wu19e) and replace the linear layers with bitlinear, either with mean or median weight measure.

## Usage

1. Install bitlinear as a package, and dependencies pytorch and pytorch-geometric
2. `python3 node_classification.py`

## Results

| **Model**                      | **Cora**  | **CiteSeer** | **PubMed** |   **Avg.**|
|:-------------------------------|------:|---------:|-------:|------:|
| SGC baseline, k=2, lr=0.001    | 75.80 | 63.20    | 76.20  | 71.73 |
| SGC baseline, k=2, lr=0.01     | 77.40 | 63.40    | 75.70  | 72.17 |
| SGC-1.58b-mean, k=2, lr=0.01   | 77.50 | 60.90    | 76.00  | 71.47 |
| SGC-1.58b-median, k=2, lr=0.01 | 78.20 | 65.50    | 74.80  | 72.88 |

