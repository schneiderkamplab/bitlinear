# Xor example


We build a dataset, where the task is to learn the xor function of two inputs.
To make it interesting, we add noise inputs.
We want to see if BitNet-MLP models (1 hidden layer) are able to ignore the noise.
All BitNet hyperparameters are fixed, except the weight measure function. Most importantly, the weight range is set to 1.58 bits (-1, 0, or 1).
We train until convergence: 1k epochs on 5k examples with 4 features, two of which determine the desired output, the other two are noise.


## Weight measure AbsMax

When trained with weight measure AbsMax, BitNet finds a solution (100% accuracy) with aminimal amount of nonzero input features.

```
tensor([[ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [ 1., -1.],
        [-1.,  1.],
        [ 0.,  0.],
        [ 0.,  0.]], grad_fn=<IndexBackward0>)
Input2hidden weights cols 0-1 L1 norm: 4.0

Input2hidden weights cols 2-3 (should be zero)
tensor([[0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]], grad_fn=<IndexBackward0>)
Input2hidden weights cols 2-3 L1 norm: 0.0
```

## Weight measure AbsMean

With weight measure AbsMean, BitNet finds a solution too, yet with slightly more nonzero parameters (7 instead of the minimum 4).

```
Input2hidden weights cols 0-1 (should be xor)
tensor([[ 0.,  0.],
        [-1.,  1.],
        [ 1., -1.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1.,  1.],
        [ 0.,  0.],
        [-1.,  0.]], grad_fn=<IndexBackward0>)
Input2hidden weights cols 0-1 L1 norm: 7.0
---
Input2hidden weights cols 2-3 (should be zero)
tensor([[0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]], grad_fn=<IndexBackward0>)
Input2hidden weights cols 2-3 L1 norm: 0.0
```


## Weight measure AbsMedian

AbsMedian-Bitnet did not find a solution with 8 bits, converging at accuracy 87.9%.
L1 norm of the xor input weights was 11 and L1 norm of the noise input weights was 7.
Retry with double hidden units (16).

With 16 hidden units, AbsMedian-BitNet found a solution but overfitted, ending again at 87.9%. The L1
norm for xor input weights was 28. The L1 norm for noise input weights was 15.

With 32 hidden units, AbsMedian-BitNet found a solution and kept it (100% accuracy). 
The L1 norm of the xor-input weights was 53.
The L1 norm of the noise-input weights was 27. 

Going back to 8 hidden units, but larger learning rate (0.1),
the model gets to 100% accuracy. L1 norm xor: 12, L1 norm noise: 6 (with 4 out of 6 being negative)
The hidden to output layer had all non-zero weights (L1 norm of 16).
```
Input2hidden weights cols 0-1 (should be xor)
tensor([[-1., -1.],
        [ 0.,  0.],
        [-1.,  1.],
        [ 1., -1.],
        [ 1., -1.],
        [ 1.,  1.],
        [ 0.,  0.],
        [-1.,  1.]])
Input2hidden weights cols 0-1 L1 norm: 12.0

Input2hidden weights cols 2-3 (should be zero)
tensor([[ 1.,  0.],
        [-1.,  1.],
        [ 0.,  0.],
        [-1.,  0.],
        [ 0.,  0.],
        [-1., -1.],
        [ 0.,  0.],
        [ 0.,  0.]])
Input2hidden weights cols 2-3 L1 norm: 6.0

Output layer:
tensor([[ 1.,  1., -1., -1., -1.,  1.,  1., -1.],
        [-1., -1.,  1.,  1.,  1., -1., -1.,  1.]])
```



## Non-BitNet MLP for comparison

Same specs as above, 8 hidden units.

```
Input2hidden weights cols 0-1 L1 norm: 30.636106491088867 # xor inputs
Input2hidden weights cols 2-3 L1 norm: 1.2088327407836914 # noise inputs
```