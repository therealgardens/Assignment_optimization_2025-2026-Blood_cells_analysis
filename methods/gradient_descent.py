import numpy as np

def softmax(A, X):
    Z = np.dot(A, X)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy_loss(A, X, b):
    m = A.shape[0]
    prob = softmax(A, X)
    correct_prob = prob[np.arange(m), b]
    loss = -np.mean(np.log(correct_prob + 1e-15)) 
    return loss

def compute_gradient(A, X, b, k):
    m = A.shape[0]
    probabilities = softmax(A, X)
    
    indicator = np.zeros((m, k))
    indicator[np.arange(m), b] = 1
    
    grad = np.dot(A.T, (probabilities - indicator)) / m
    return grad

def gradient_descent(A, b, k, tol = 0.000001, max_iter = 1000, stepsize_method = "constant"):
    m = A.shape[0]
    d = A.shape[1]
    spectral_norm = np.linalg.norm(A, ord = 2)
    L = (1/(4*m))*(spectral_norm**2)
    #choosing starting point 
    X = np.zeros((d, k))
    for i in range(max_iter):
        grad = compute_gradient(A, X, b, k)
        gradient_norm = np.linalg.norm(grad)
        if gradient_norm < tol:
            print(f"converged in {i} iterations.")
            return X
        
        #stepsize selection criterion 
        if stepsize_method == "constant":
            alpha = 1/L
        elif stepsize_method == "armijo":
            #initialising 
            alpha = 1.0
            sigma = 0.0001
            rho = 0.5
            current_loss = cross_entropy_loss(A, X, b)
            while cross_entropy_loss(A, X - alpha*grad, b) > current_loss - sigma*alpha*(gradient_norm**2):
                alpha = alpha*rho

        #update weights
        X = X - alpha * grad
        if i % 100 == 0:
            print(f"iteration {i}: gradient Norm = {gradient_norm:.4f}")
            
    return X
