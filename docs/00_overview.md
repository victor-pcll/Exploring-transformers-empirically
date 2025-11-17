# Overview

This project implements a full experimental pipeline for studying histogram prediction
from attention-like neural network models.  

The code generates synthetic datasets, trains a neural network model to predict 
token occurrence histograms, evaluates the performances, and stores all results
for later statistical analysis.

This documentation summarizes the structure of the codebase and explains the logic
of each module:

- How datasets are generated
- How the neural network produces attention matrices and histogram predictions
- How the training loop works
- How experiments are orchestrated and logged