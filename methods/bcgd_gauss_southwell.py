import time
import numpy as np

def gs_bcgd(x_init, oracle, block_step_size, max_iter = 5000, debug_logs = False, acc_100 = False):

    k = 1
    x_k = x_init.copy()
    start_time = time.time() 

    while k < max_iter:
        #GS selection rule
        full_grad = oracle(x_k) #gets the gradient for the whole X
        block_norms = np.linalg.norm(full_grad, axis=1) #calculates the norm of each block (row)

        i_star = np.argmax(block_norms) #selects the block with the highest gradient norm

        #Update
        x_k_1 = x_k.copy()
        x_k_1[i_star, :] = x_k[i_star, :] - block_step_size[i_star] * full_grad[i_star, :] #Updates only on the selected block

        if acc_100 == True:
            if accuracy(oracle.predictors, oracle.predictions, x_k_1) == 1:
                print("Accuracy of 100% reached")
                break

        if debug_logs == True:

            if k % 500 == 0:
                acc = accuracy(oracle.predictors, oracle.predictions, x_k_1)
                print(f"iteration: {k}, selected_block: {i_star}, sequential distance: {np.linalg.norm(x_k - x_k_1)}, loss: {oracle.loss(x_k_1):.6f}")
                print(f"accuracy: {acc*100:.2f}%")

        last_dist = np.linalg.norm(x_k - x_k_1)
        x_k = x_k_1
        k += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Duration: {elapsed_time:.2f} s")
    print(f"\nFinal iteration: {k}, selected_block: {i_star}, sequential distance: {last_dist}, loss: {oracle.loss(x_k):.4f}")
    print(f"accuracy: {accuracy(oracle.predictors, oracle.predictions, x_k)*100:.2f}%\n")
    
    return x_k
