import numpy as np

rng = np.random.default_rng(seed=42)

class MultiLogisticRegressionOracle:
    def __init__(self, predictors: np.array, predictions: list, n_categories: int):
        self.predictors = predictors # Referenced as A
        self.predictions = predictions # Referenced as b

        self.n_features = predictors.shape[1] # Referenced as d
        self.n_categories = n_categories # Referenced as k

        self.indicator = np.array([predictions == c for c in range(self.n_categories)]).T
        self.indicator = self.indicator.astype(int) # m x k
    
    def loss(self, x: np.array):
        result = 0
        for i in range(len(self.predictors)):
            result -= np.dot(x[:,self.predictions[i]], self.predictors[i])
            result += np.log(np.sum(np.exp(np.dot(self.predictors, x))))
        
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
            result = np.dot(self.predictors.T, self.indicator - softmax_entries)

        return result


def accuracy(predictors: np.array, ground_truth:np.array, regresssors: np.array) -> float:
    predictions = np.argmax(np.dot(predictors, regresssors), axis=1)
    return np.sum(predictions == ground_truth)/len(ground_truth)
