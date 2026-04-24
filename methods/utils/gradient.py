import numpy as np

class MultiLogisticRegressionOracle:
    def __init__(self, predictors: np.array, predictions: np.array, n_categories: int):
        self.predictors = predictors # Referenced as A
        self.predictions = predictions # Referenced as b

        self.n_features = predictors.shape[1] # Referenced as d
        self.n_categories = n_categories # Referenced as k

        self.n_blocks = self.n_features

        self.indicator = np.array([predictions == c for c in range(self.n_categories)]).T
        self.indicator = self.indicator.astype(int) # m x k
    
    def loss(self, x: np.array):
        scores = np.dot(self.predictors, x) # m x k 
        exp = np.exp(scores)
        sum_exp = np.sum(exp, axis=1)
        log_sum_exp = np.log(sum_exp)
    
        class_scores = scores[range(len(self.predictions)), self.predictions]
        result = -np.sum(class_scores) + np.sum(log_sum_exp)
        return result

    def __call__(self, x: np.array, j_block: int|None = None):
        """Estimate the gradient in regard of specific categories."""
        transformed_entries = np.exp(np.dot(self.predictors, x)) # m x k
        summed_entries = np.sum(transformed_entries, axis=1, keepdims=True)
        softmax_entries = transformed_entries / summed_entries # m x k

        if j_block is not None:
            result = np.zeros((self.n_features, self.n_categories)) # d x k

            result[j_block] = -np.dot(self.predictors.T[j_block,:], self.indicator - softmax_entries)
        else:
            result = -np.dot(self.predictors.T, self.indicator - softmax_entries)

        return result


class CachedMultiLogisticRegressionOracle(MultiLogisticRegressionOracle):
    def __init__(self, predictors, predictions: np.array, n_categories):
        super().__init__(predictors, predictions, n_categories)
        self.S = np.zeros((self.predictors.shape[0], self.n_categories)) #Initializing scores AX (m x k)

    def update_cache(self, i_star: int, delta_x_row):
        self.S += np.outer(self.predictors[:, i_star], delta_x_row)
    
    def __call__(self):
        exp_S = np.exp(self.S)
        summed_entries = np.sum(exp_S, axis=1, keepdims=True)
        softmax_entries = exp_S / summed_entries

        result = -np.dot(self.predictors.T, self.indicator - softmax_entries)

        return result
