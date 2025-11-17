# Dataset Module

The dataset used in this project consists of sequences of discrete tokens and
their corresponding histograms.

## HistogramDataset

Located in: `histo_tied_exp_cluster.py`

### Purpose
Generate random sequences and compute for each the vector of token counts
(histogram).

### Main attributes
- `seq_len`: length of each sequence
- `L`: alphabet size
- `n_samples`: total number of generated sequences
- `X`: matrix of sequences, shape `(N_total, seq_len)`
- `y`: matrix of histograms, shape `(N_total, seq_len)`

### How histograms are computed
The `hist(s)` function counts occurrences of each token inside a sequence and
returns a count for each position.

### Splitting into train/test sets
The helper function `prepare_dataset(config)` initializes a full dataset and
splits it using `torch.utils.data.random_split`.