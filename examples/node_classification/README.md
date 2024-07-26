# Node Classification Example

In node classification, the model must learn a mapping from node features $X$ and node connectivity $A$ to class labels $y$.
We experiment with a simple yet powerful model for node classification: Simplified Graph Convolution (SGC).



| **Model**                      | **Cora**  | **CiteSeer** | **PubMed** |   **Avg.**|
|:-------------------------------|------:|---------:|-------:|------:|
| SGC, k=2, lr=0.001             | 75.80 | 63.20    | 76.20  | 71.73 |
| SGC-1.58b-mean, k=2, lr=0.01   | 77.50 | 60.90    | 76.00  | 71.47 |
| SGC-1.58b-median, k=2, lr=0.01 | 78.20 | 65.50    | 74.80  | 72.88 |

