import numpy as np


def distri_powermethod(uinit, multipliedmat, NodeN, networkgraph, powerIterations, consensusIterations):
    
    """
    Parameters
    ----------
    Inputs:
        uinit: initial random eigenvector
        multipliedmat: 3D array containing matrices E_R E_R^T from each site
        NodeN: number of nodes/sites in network
        networkGraph: a doubly stochastic matrix corresponding to the graph representing the network topology
        powerIterations: fixed number of power method iterations
        consensusIterations: fixed number of iterations for distributed summation
    Outputs:
        unew: updated dictionary atom
    """
    
    uinitmat = np.tile(np.copy(uinit), (NodeN, 1))
    
    for poweriter in range(powerIterations):
        localvectorPowerMethod = np.zeros([NodeN, multipliedmat.shape[0]])
        
        for nodes in range(NodeN):
            localvectorPowerMethod[nodes,:] = np.dot(multipliedmat[:,:,nodes], uinitmat[nodes,:])
            
        summatrix = np.copy(localvectorPowerMethod)
        
        for consiter in range(consensusIterations):
            summatrix = np.dot(networkgraph, summatrix)
            
        unew = np.zeros(uinitmat.shape)
        for nodes in range(NodeN):
            unew[nodes,:] = summatrix[nodes,:]/np.linalg.norm(summatrix[nodes,:])
            
    return unew