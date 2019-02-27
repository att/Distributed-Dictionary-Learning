import time
import numpy as np
from sparsecoding import sparse_encode_omp
from sparsecoding import sparse_encode_nnmp
from dictionaryupdate import l2_update_dict
from dictionaryupdate import nnl2_update_dict


def dict_learning_ksvd(X, dict_init, s, max_iter):
    
    """Solves a dictionary learning matrix factorization problem.
    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving:
        (D, Theta) = argmin || X - D Theta ||_F^2 
                    with || d_k ||_2 = 1 and || theta_i ||_0 <= s
    where D is the dictionary and Theta is the sparse code.
    Parameters
    ----------
    Inputs:
        X: data matrix
        dict_init: array of shape (n_features, n_components)
        s: sparsity controlling parameter
        max_iter: maximum number of iterations to perform
    Outputs:
        D: dictionary at every iteration
    """
    
    sigdim = X.shape[0]
    D = np.zeros([sigdim, dict_init.shape[1], max_iter+1])
    D[:,:,0] = dict_init
    
    for iters in range(max_iter):
        t0 = time.time()
        
        # Update coefficients
        coef = sparse_encode_omp(X, D[:,:,iters], s)
        # Update dictionary
        D[:,:,iters+1], coef = l2_update_dict(D[:,:,iters], X, coef)
        dt = time.time() - t0
        print('the %dth iteration takes %f seconds' %(iters,dt))
        
    return D
    
    
def dict_learning_nnksvd(X, dict_init, s, max_iter, updatec_iter):
    
    """Solves a dictionary learning matrix factorization problem.
    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving:
        (D, Theta) = argmin || X - D Theta ||_F^2 
                    with D, Theta >= 0, || d_k ||_2 = 1 and || theta_i ||_0 <= s
    where D is the dictionary and Theta is the sparse code.
    Parameters
    ----------
    Inputs:
        X: data matrix
        dict_init: array of shape (n_features, n_components)
        s: sparsity controlling parameter
        max_iter: maximum number of iterations to perform
        updatec_iter: number of iterations for updating coefficients in the sparse coding
    Outputs:
        D: dictionary at every iteration
    """
    
    sigdim = X.shape[0]
    D = np.zeros([sigdim, dict_init.shape[1], max_iter+1])
    D[:,:,0] = dict_init
    
    for iters in range(max_iter):
        t0 = time.time()
        
        # Update coefficients
        coef = sparse_encode_nnmp(X, D[:,:,iters], s, updatec_iter)
        # Update dictionary
        D[:,:,iters+1], coef = nnl2_update_dict(D[:,:,iters], X, coef)
        dt = time.time() - t0
        print('the %dth iteration takes %f seconds' %(iters,dt))
        
    return D