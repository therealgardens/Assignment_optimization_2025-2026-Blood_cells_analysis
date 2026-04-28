from typing import Callable

import time
import numpy as np

from methods.utils.accuracy import accuracy


def gs_bcgd(
    x_init: np.array,
    oracle: Callable,
    step_size: list[float],
    tol: float = 0.000001, 
    max_iter: int = 10000,
    debug: bool = False,
)-> dict:
    x_k = x_init.copy()
    iteration_end_times = [0.0]
    iteration_history = []

    start_time = time.process_time()
    for k in range(max_iter):
        #GS selection rule
        full_grad = oracle() #gets the gradient for the whole X
        block_norms = np.linalg.norm(full_grad, axis=1) #calculates the norm of each block (row)

        i_star = np.argmax(block_norms) #selects the block with the highest gradient norm

        #Update
        x_k_1 = x_k.copy()

        delta_x_row = -step_size[i_star]*full_grad[i_star, :]

        x_k_1[i_star, :] = x_k[i_star, :] + delta_x_row
        
        oracle.update_cache(i_star, delta_x_row)
        
        if debug and k % 500 == 0:
            acc = accuracy(oracle.predictors, oracle.predictions, x_k_1)
            print(f"iteration: {k}, selected_block: {i_star}, sequential distance: {np.linalg.norm(x_k - x_k_1)}, loss: {oracle.loss(x_k_1):.6f}")
            print(f"accuracy: {acc*100:.2f}%")

        last_dist = np.linalg.norm(x_k - x_k_1)
        x_k = x_k_1

        iteration_history.append(x_k)
        iteration_end_times.append(time.process_time()-start_time)

        if last_dist < tol:
            break
    
    return {
        "iterations": k,
        "x_k": x_k,
        "history": iteration_history,
        "times": iteration_end_times,
    }
