# Training Loop

The main training function is `train_student_on_data`.

### Goal
Train a student model to predict normalized histograms from sequences.

### Training procedure
1. Load full training batch.
2. Compute:
   - attention matrix
   - predicted histogram
3. Compute loss:
   - data loss: MSE between predicted and true normalized histograms
   - regularization loss: L2 penalty on weight matrix
4. Update weights with Adam optimizer.
5. Repeat for T iterations or until convergence.

### Outputs
Returned values include:
- `W_student`: learned weight matrix
- `data_loss_final`
- `reg_loss_final`
- `attn_matrix_final`: attention matrix for a sample sequence
- `seq_sample`: corresponding sequence