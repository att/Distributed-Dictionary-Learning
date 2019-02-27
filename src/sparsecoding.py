import numpy as np


def omp(D, x, s):
    
    """Orthogonal matching pursuit (OMP)
    
    Solves min || D * theta - x ||_2 subject to || theta ||_0 <= s
    
    Parameters
    ----------
        D: input dictionary
        x: vector of length n_features
        s: number of nonzero coefficients
    """
    
    residual = x
    idx = []
    
    while len(idx)<s:
        lam = np.abs(np.dot(D.T, residual)).argmax()
        idx.append(lam)
        gamma, _, _, _ = np.linalg.lstsq(D[:,idx], x)
        residual = x - np.dot(D[:,idx], gamma)
        
    theta = np.zeros(D.shape[1])
    theta[idx] = gamma
    return theta, idx
    
    
def sparse_encode_omp(X, D, sparsity):
    K = D.shape[1]
    Theta = np.zeros([K, X.shape[1]])
    for i in range(X.shape[1]):
        Theta[:,i], idx = omp(D, X[:,i], sparsity)
        
    return Theta
    
    
    
def nnmp(D, x, s, citer):
    
    """Nonnegative matching pursuit (NMP)
    
    Solves min || D * theta - x ||_2 subject to || theta ||_0 <= s and theta >= 0
    
    Parameters
    ----------
        D: input dictionary
        x: vector of length n_features
        s: number of nonzero coefficients
        citer: number of iterations for updating coefficients
    """
    
    residual = x
    idx = []
    gamma = np.asarray([])
    
    while len(idx)<s:
        val = np.dot(D.T, residual).max()
        if val<=0:
            break
        else:
            lam = np.dot(D.T, residual).argmax()
            gamma = np.append(gamma, val)
            idx.append(lam)
            for ii in range(citer):
                gamma = gamma*np.dot(D[:,idx].T, x)/np.dot(np.dot(D[:,idx].T, D[:,idx]), gamma)
                
            residual = x - np.dot(D[:,idx], gamma)
            
    theta = np.zeros(D.shape[1])
    theta[idx] = gamma
    return theta, idx
    
    
def sparse_encode_nnmp(X, D, sparsity, updatec_iter):
    K = D.shape[1]
    Theta = np.zeros([K, X.shape[1]])
    for i in range(X.shape[1]):
        Theta[:,i], idx = nnmp(D, X[:,i], sparsity, updatec_iter)
        
    return Theta