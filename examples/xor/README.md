# Xor example


We build a dataset, where the task is to learn the xor function of two inputs.
To make it interesting, we add noise inputs.
We want to see if BitNet-MLP models (one input2hidden and one hidden2output layer) are able to ignore the noise.
All BitNet hyperparameters are fixed, except the weight measure function. Most importantly, the weight range is set to 1.58 bits (-1, 0, or 1).
We train until convergence: 1k epochs on 5k examples with 4 features, two of which determine the desired output, the other two are noise.
Learning rate is set to 0.01. Hidden size is 8 unless noted otherwise.




## mlp-1.58b-mean

With weight measure AbsMean, BitNet finds a perfect solution (100% accuracy) with 4 nonzero parameters on the input layer.

```
Input2hidden weights cols 0-1 (should be xor)
tensor([[ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1.,  1.],
        [ 0.,  0.],
        [ 1., -1.]])
Input2hidden weights cols 0-1 L1 norm: 4.0

Input2hidden weights cols 2-3 (should be zero)
tensor([[0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]])
Input2hidden weights cols 2-3 L1 norm: 0.0
```

## mlp-1.58b-median


With AbsMedian scaling, mlp-1.58-median did not find a solution with 8 bits, converging at accuracy 86.8%.
L1 norm of the xor input weights was 12 and L1 norm of the noise input weights was 8.
Let's retry with double hidden units (16).

With 16 hidden units, mlp-1.68b-median ended at perfect accuracy, although trajectory was unstable.
The L1 norm for xor input weights was 24. The L1 norm for noise input weights was 17. Output layer L1 norm 26 (out of 32).

With 32 hidden units, mlp-1.68b-median found a 100% accuracy solution with less fluctuation on the trajectory.
The L1 norm of the xor-input weights was 51.
The L1 norm of the noise-input weights was 28. 
The L1 norm of the output layer was 50 (out of 64).

Going back to 8 hidden units, but larger learning rate (0.1),
the model gets to 100% accuracy. L1 norm xor: 12, L1 norm noise: 5 (with 4 out of 5 being negative)
The hidden to output layer had all non-zero weights (L1 norm of 16).

The bias parameters seemingly help to ignore parts of the inputs

TLDR; The model does not manage to assign zero input weight to noise inputs, but most remaining ones are negative (at least with lr=0.1) and thus likely get ignored via ReLU. Bias parameters also help to compensate.

Details below:

```
Input2hidden weights cols 0-1 (should be xor)
tensor([[ 0.,  0.],
        [ 1., -1.],
        [-1., -1.],
        [-1.,  1.],
        [ 1., -1.],
        [ 0.,  0.],
        [-1.,  1.],
        [ 1.,  1.]])
Input2hidden weights cols 0-1 L1 norm: 12.0

Input2hidden weights cols 2-3 (should be zero)
tensor([[ 0.,  0.],
        [ 0.,  0.],
        [-1.,  1.],
        [ 0.,  0.],
        [-1.,  0.],
        [ 0.,  0.],
        [ 0.,  0.],
        [-1., -1.]])
Input2hidden weights cols 2-3 L1 norm: 5.0
```

```
tensor([[ 1., -1.,  1., -1., -1.,  1., -1.,  1.],
        [-1.,  1., -1.,  1.,  1., -1.,  1., -1.]])
hidden2output weights L1 norm: 16.0
```


```
Input 2 hidden bias:
tensor([-1.1960e-20, -4.0664e+00,  2.9069e+00, -1.2000e+00, -2.4540e+00,
        -2.8123e-11, -4.3038e+00, -2.8608e+00], requires_grad=True)

Hidden 2 output bias:
tensor([ 2.3987, -2.3987], requires_grad=True)
```