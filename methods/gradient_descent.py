from typing import Callable
import time

import numpy as np

from methods.utils.accuracy import accuracy


def gradient_descent(
    x_init: np.array, 
    oracle: Callable, 
    step_size: float, 
    tol: float = 0.000001, 
    max_iter: int = 10000,
    debug: bool = False,
) -> dict:
    x_k = x_init
    iteration_end_times = []
    iteration_history = []

    start_time = time.process_time()
    for k in range(max_iter):
        gradient = oracle(x_k)
        gradient_norm = np.linalg.norm(gradient)

        x_k_1 = x_k - step_size * gradient
        
        if debug and k % 500 == 0:
            current_loss = oracle.loss(x_k_1)
            current_acc = accuracy(oracle.predictors, oracle.predictions, x_k_1)
            print(f"Iteration {k}: Loss = {current_loss}, sequential distance: {np.linalg.norm(x_k-x_k_1)}, Accuracy = {current_acc}")
            
        x_k = x_k_1

        iteration_history.append(x_k)
        iteration_end_times.append(time.process_time()-start_time)

        if step_size * gradient_norm < tol:
            break

    return {
        "iterations": k,
        "x_k": x_k,
        "history": iteration_history,
        "times": iteration_end_times,
    }
