import numpy as np

def softmax(A, X):
    Z = np.dot(A, X)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def compute_gradient(A, X, b, k):
    m = A.shape[0]
    probabilities = softmax(A, X)
    
    indicator = np.zeros((m, k))
    indicator[np.arange(m), b] = 1
    
    grad = np.dot(A.T, (probabilities - indicator)) / m
    return grad

def gradient_descent(A, b, k, tol = 0.000001, max_iter = 1000):
    m = A.shape[0]
    d = A.shape[1]
    spectral_norm = np.linalg.norm(A, ord = 2)
    L = (1/(4*m))*(spectral_norm**2) 
    alpha = 1/L
    #choosing starting point 
    X = np.zeros((d, k))
    for i in range(max_iter):
        grad = compute_gradient(A, X, b, k)
        gradient_norm = np.linalg.norm(grad)
        if gradient_norm < tol:
            print(f"converged in {i} iterations.")
            return X
        #update weights
        X = X - alpha * grad
        if i % 100 == 0:
            print(f"iteration {i}: gradient Norm = {gradient_norm:.4f}")
            
    return X

