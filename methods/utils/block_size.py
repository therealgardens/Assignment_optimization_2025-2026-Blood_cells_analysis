import numpy as np

def block_step_size(A):
  
    column_norms_sq = np.sum(A**2, axis=0)
    block_L = 0.25 * column_norms_sq
    block_step_size = 1.0 / block_L
    block_step_size[block_step_size == float("inf")] = 1.0 # Just in case of zero columns
    block_step_size[block_step_size == -float("inf")] = 1.0

    return block_step_size

def full_step_size(A):
  
    norm_sq = np.sum(A**2)
    return 1 / (0.25 * norm_sq)
