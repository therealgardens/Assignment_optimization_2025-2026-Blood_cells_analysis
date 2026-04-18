import numpy as np

def block_step_size(A):
  
    column_norms_sq = np.sum(A**2, axis=0)
    block_L = 0.25 * column_norms_sq
    block_step_size = 1.0 / block_L

    return block_step_size
