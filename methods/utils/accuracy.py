import numpy as np

def accuracy(predictors: np.array, ground_truth:np.array, regresssors: np.array) -> float:
    predictions = np.argmax(np.dot(predictors, regresssors), axis=1)
    return np.sum(predictions == ground_truth)/len(ground_truth)
