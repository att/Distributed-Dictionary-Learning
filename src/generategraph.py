import numpy as np
from breathfirstsearch import bfs


def gengraph(N, p):
    
    A = np.random.rand(N,N)
    A = np.triu(A,1)
    A = A + A.T
    networkGraph = (A<p).astype(float)
    
    spt, b, found, path = bfs(networkGraph, 0, N-1)
    while min(b)<0:
        A = np.random.rand(N,N)
        A = np.triu(A,1)
        A = A + A.T
        networkGraph = (A<p).astype(float)
        spt, b, found, path = bfs(networkGraph, 0, N-1)
        
    for i in range(N):
        networkGraph[i,i] = 0
    
    temp = np.copy(networkGraph)
    
    for i in range(N):
        for j in range(N):
            networkGraph[i,j] = 1/max(np.sum(temp[i,:]),np.sum(temp[j,:]))*temp[i,j]
            
    for i in range(N):
        networkGraph[i,i] = 1 - np.sum(networkGraph[i,:])
        
    return networkGraph