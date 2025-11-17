# Neural Network Model

The model is defined in the `Net` class.

Its goal is:
1. Embed each token into a D-dimensional vector.
2. Apply a linear transformation.
3. Compute a self-attention matrix using inner products.
4. Apply a softmax row-wise.
5. Sum each row to obtain histogram predictions.

## Architecture

### Layers
- `Embedding(L, D)`
- `Linear(D → R)` with no bias
- Attention computed with:
$$attention[i,j] = (x_i · x_j) / sqrt(R)$$

### Output
- `x`: attention matrix of shape `(N, seq_len, seq_len)`
- `h_pred`: histogram prediction obtained by summing softmax rows

### Noise injection
If `delta_in > 0`, optional symmetric Gaussian noise is added to the attention.

### Hyperparameters
- `D`: input embedding dimension
- `R`: hidden dimension
- `L`: vocabulary size
- `beta`: scaling parameter for attention sharpness
- `norm_init`: initialization of the linear weights