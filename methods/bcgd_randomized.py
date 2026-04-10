from typing import Callable

import numpy as np

rng = np.random.default_rng(seed=42)

class MultiLogisticRegressionOracle:
    def __init__(self, predictors: np.array, predictions: list):
        self.predictors = predictors # Referenced as A
        self.predictions = predictions # Referenced as b

        self.n_features = predictors.shape[1] # Referenced as d
        self.n_categories = np.max(predictions) + 1 # Referenced as k

        self.indicator = np.array([predictions == c for c in range(self.n_categories)])
        self.indicator = self.indicator.astype(int) # m x k

    def __call__(self, x: np.array, cat_interval: tuple[float]|None = None):
        """Estimate the gradient in regard of specific categories."""
        transformed_entries = np.exp(np.dot(self.predictors, x)) # m x k
        summed_entries = np.sum(transformed_entries, axis=1, keepdims=True)
        softmax_entries = transformed_entries / summed_entries # m x k

        active_gradient = self.indicator[:, cat_interval[0]:cat_interval[1]] # d x |b_i|
        active_gradient -= softmax_entries[:, cat_interval[0]:cat_interval[1]]

        result = np.zeros((self.n_features, self.n_categories)) # d x k
        result[:, cat_interval[0]:cat_interval[1]] = active_gradient

        return result


def randomized_bcgd(
    x_init: np.array,
    block_markers: list[int],
    oracle: Callable,
    alpha: float
) -> np.array:
    """Estimate BCGD optimization with Randomized Rule."""
    n_blocks = len(block_markers)

    k = 1
    x_k = x_init

    while True:
        # TODO: Define if its worth it to implement other distributions
        i_k = rng.integers(1, n_blocks, 1)[0]
        category_range = (block_markers[i_k-1], block_markers[i_k])

        g_i_k = oracle(x_k, category_range)
        x_k_1 = x_k - alpha * g_i_k

        if ...: # TODO: Define stop condition
            break

        x_k = x_k_1
        k += 1

    return (x_k, k)
