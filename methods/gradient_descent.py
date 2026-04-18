from typing import Callable
import numpy as np
from utils.gradient import accuracy, MultiLogisticRegressionOracle


# def softmax(A, X):
#     Z = np.dot(A, X)
#     Z = Z - np.max(Z, axis=1, keepdims=True)
#     exp_Z = np.exp(Z)
#     return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# def cross_entropy_loss(A, X, b):
#     m = A.shape[0]
#     prob = softmax(A, X)
#     correct_prob = prob[np.arange(m), b]
#     loss = -np.mean(np.log(correct_prob + 1e-15)) 
#     return loss

# def compute_gradient(A, X, b, k):
#     m = A.shape[0]
#     probabilities = softmax(A, X)
    
#     indicator = np.zeros((m, k))
#     indicator[np.arange(m), b] = 1
    
#     grad = np.dot(A.T, (probabilities - indicator)) / m
#     return grad

# def gradient_descent(A, b, k, tol = 0.000001, max_iter = 1000, stepsize_method = "constant"):
#     m = A.shape[0]
#     d = A.shape[1]
#     spectral_norm = np.linalg.norm(A, ord = 2)
#     L = (1/(4*m))*(spectral_norm**2)
#     #choosing starting point 
#     X = np.zeros((d, k))
#     for i in range(max_iter):
#         grad = compute_gradient(A, X, b, k)
#         gradient_norm = np.linalg.norm(grad)
#         if gradient_norm < tol:
#             print(f"converged in {i} iterations.")
#             return X
        
#         #stepsize selection criterion 
#         if stepsize_method == "constant":
#             alpha = 1/L
#         elif stepsize_method == "armijo":
#             #initialising 
#             alpha = 1.0
#             sigma = 0.0001
#             rho = 0.5
#             current_loss = cross_entropy_loss(A, X, b)
#             while cross_entropy_loss(A, X - alpha*grad, b) > current_loss - sigma*alpha*(gradient_norm**2):
#                 alpha = alpha*rho

#         #update weights
#         X = X - alpha * grad
#         if i % 100 == 0:
#             print(f"iteration {i}: gradient Norm = {gradient_norm:.4f}")
            
#     return X


###attempt with implementation of oracle however doesnt work well when i try it 


def gradient_descent(
    x_init: np.array, 
    oracle: Callable, 
    alpha: float, 
    tol: float = 0.000001, 
    max_iter: int = 10000
    ) -> tuple[np.array, int]:
    
    x_k = x_init
    for k in range(1, max_iter + 1):
        gradient = oracle(x_k)
        gradient_norm = np.linalg.norm(gradient)

        if k == 1000000:# gradient_norm < tol:
            print(f"Converged at iteration {k}. Gradient Norm: {gradient_norm}")
            break
        x_k_1 = x_k - alpha * gradient
        
        if k % 5 == 0:
            current_loss = oracle.loss(x_k_1)
            current_acc = accuracy(oracle.predictors, oracle.predictions, x_k_1)
            print(f"Iteration {k}: Loss = {current_loss}, Accuracy = {current_acc}")
            
        x_k = x_k_1

    return (x_k, k)




rng = np.random.default_rng(seed=42)
m = 1000
d = 1000
k = 50  

A = rng.random((m, d))
X = rng.random((d, k))
E = rng.random((m, k))


scores = np.dot(A, X) + E
b = np.argmax(scores, axis = 1) 
oracle = MultiLogisticRegressionOracle(predictors=A, predictions=b, n_categories=k)

x_init = np.zeros((d, k))


spectral_norm = np.linalg.norm(A, ord=2)
L = (1 / (4 * m)) * (spectral_norm**2)
safe_alpha = (1 / L)/100

final_weights, total_steps = gradient_descent(
    x_init=x_init,
    oracle=oracle,
    alpha=safe_alpha)

