from typing import Callable

import numpy as np

from methods.utils.gradient import accuracy

rng = np.random.default_rng(seed=42)

def randomized_bcgd(
    x_init: np.array,
    n_categories: int,
    oracle: Callable,
    alpha: float
) -> np.array:
    """Estimate BCGD optimization with Randomized Rule."""
    k = 1
    x_k = x_init

    while True:
        i_k = rng.integers(0, n_categories, 1)[0]

        g_i_k = oracle(x_k, i_k)
        x_k_1 = x_k - alpha * g_i_k

        if k >= 100000: # TODO: Define stop condition
            break

        if k%500 == 0: # Just for debugging purposes
            print(f"iteration: {k}, sequential distance: {np.linalg.norm(x_k-x_k_1)}, loss: {oracle.loss(x_k_1)}")
            print(f"accuracy: {accuracy(oracle.predictors, oracle.predictions, x_k_1)}")
        
        x_k = x_k_1
        k += 1

    return (x_k, k)
