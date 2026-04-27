import time
from typing import Callable

import numpy as np

from methods.utils.accuracy import accuracy


rng = np.random.default_rng(seed=42)

def randomized_bcgd(
    x_init: np.array,
    oracle: Callable,
    step_size: list[float],
    tol: float = 0.000001, 
    max_iter: int = 10000,
    debug: bool = False,
) -> dict:
    """Estimate BCGD optimization with Randomized Rule."""
    x_k = x_init
    iteration_end_times = []
    iteration_history = []

    start_time = time.process_time()
    for k in range(max_iter):
        i_k = rng.integers(0, oracle.n_blocks, 1)[0]

        g_i_k = oracle(x_k, i_k)
        x_k_1 = x_k - step_size[i_k] * g_i_k

        if debug and k%500 == 0: # Just for debugging purposes
            print(f"iteration: {k}, sequential distance: {np.linalg.norm(x_k-x_k_1)}, loss: {oracle.loss(x_k_1)}")
            print(f"accuracy: {accuracy(oracle.predictors, oracle.predictions, x_k_1)}")
        
        x_k = x_k_1

        iteration_history.append(x_k)
        iteration_end_times.append(time.process_time()-start_time)

        if np.linalg.norm(step_size[i_k] * g_i_k) < tol:
            break

    return {
        "iterations": k,
        "x_k": x_k,
        "history": iteration_history,
        "times": iteration_end_times,
    }
