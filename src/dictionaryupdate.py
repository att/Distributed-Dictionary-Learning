import numpy as np


def l2_update_dict(olddict, X, coef):
    
    """Update the dictionary for KSVD.
    
    Parameters
    ----------
    Inputs:
        olddict: dictionary in the previous iteration
        X: data matrix
        coef: sparse coding of the data against which to optimize the dictionary
    Outputs:
        newdict: updated dictionary
        coef: updated sparse coefficient matrix
    """
    
    K = coef.shape[0]
    newdict = np.copy(olddict)
    
    for k in range(K):
        indexes = np.nonzero(coef[k,:]!=0)[0]
        
        if len(indexes)>0:
            tempcoef = coef[:,indexes]
            tempcoef[k,:] = 0
            ER = X[:,indexes] - np.dot(newdict, tempcoef)
            U, s, Vh = np.linalg.svd(ER)
            newdict[:,k] = U[:,0]
            coef[k,indexes] = s[0]*Vh[0,:]
            
    return newdict, coef
    
    
def nnl2_update_dict(olddict, X, coef):
    
    """Update the dictionary for NN-KSVD.
    
    Parameters
    ----------
    Inputs:
        olddict: dictionary in the previous iteration
        X: data matrix
        coef: sparse coding of the data against which to optimize the dictionary
    Outputs:
        newdict: updated dictionary
        coef: updated sparse coefficient matrix
    """
    
    alteritern = 10
    K = coef.shape[0]
    newdict = np.copy(olddict)
    
    for k in range(K):
        indexes = np.nonzero(coef[k,:]!=0)[0]
        
        if len(indexes)>0:
            tempcoef = coef[:,indexes]
            tempcoef[k,:] = 0
            ER = X[:,indexes] - np.dot(newdict, tempcoef)
            U, s, Vh = np.linalg.svd(ER)
            u = U[:,0]
            v = s[0]*Vh[0,:]
            if np.all(v<0)==True:
                u = -u
                v = -v
                
            u = (u>0)*u
            v = (v>0)*v
            for subiters in range(alteritern):
                u = np.dot(ER, v)/np.dot(v, v)
                u = (u>0)*u
                v = np.dot(u, ER)/np.dot(u, u)
                v = (v>0)*v
                
            normu = np.linalg.norm(u)
            newdict[:,k] = u/normu
            coef[k,indexes] = normu*v
            
    return newdict, coef