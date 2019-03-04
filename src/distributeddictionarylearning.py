import time
import numpy as np
from sparsecoding import sparse_encode_omp
from sparsecoding import sparse_encode_nnmp
from distributedpowermethod import distri_powermethod


def cloud_ksvd(X, AtomN, dict_init, s, NodeN, networkGraph, vec_init, max_iter, powerIterations, consensusIterations):
    
    sigdim = X[0].shape[0]
    D = np.zeros([sigdim, AtomN, max_iter+1, NodeN])
    for i in range(NodeN):
        D[:,:,0,i] = dict_init
        
    theta = [[] for i in range(NodeN)]
    
    for iters in range(max_iter):
        t0 = time.time()
        
        for nodes in range(NodeN):
            D[:,:,iters+1,nodes] = np.copy(D[:,:,iters,nodes])
            theta[nodes] = sparse_encode_omp(X[nodes], D[:,:,iters+1,nodes], s)
            
        for k in range(AtomN):
            multipliedMat = np.zeros([sigdim,sigdim,NodeN])
            index_temp = [[] for i in range(NodeN)]
            ER_temp = [[] for i in range(NodeN)]
            
            for nodes in range(NodeN):
                indexes = np.nonzero(theta[nodes][k,:]!=0)[0]
                index_temp[nodes] = indexes
                
                if len(indexes)>0:
                    tempcoef = theta[nodes][:,indexes]
                    tempcoef[k,:] = 0
                    ER = X[nodes][:,indexes] - np.dot(D[:,:,iters+1,nodes], tempcoef)
                    ER_temp[nodes] = ER
                    multipliedMat[:,:,nodes] = np.dot(ER, ER.T)
                    
            if np.max(np.abs(multipliedMat))>0:
                newatom = distri_powermethod(vec_init, multipliedMat, NodeN, networkGraph, powerIterations, consensusIterations)
                for nodes in range(NodeN):
                    D[:,k,iters+1,nodes] = newatom[nodes,:]
                    if len(index_temp[nodes])>0:
                        theta[nodes][k,index_temp[nodes]] = np.dot(ER_temp[nodes].T, newatom[nodes,:])
                        
        dt = time.time() - t0
        print('the %dth iteration takes %f seconds' %(iters,dt))
        
    return D
    
    
def cloud_nnksvd(X, AtomN, dict_init, s, NodeN, networkGraph, vec_init, max_iter, updatec_iter, powerIterations, consensusIterations):
    
    alteritern = 10
    sigdim = X[0].shape[0]
    D = np.zeros([sigdim, AtomN, max_iter+1, NodeN])
    for i in range(NodeN):
        D[:,:,0,i] = dict_init
        
    theta = [[] for i in range(NodeN)]
    
    for iters in range(max_iter):
        t0 = time.time()
        
        for nodes in range(NodeN):
            D[:,:,iters+1,nodes] = np.copy(D[:,:,iters,nodes])
            theta[nodes] = sparse_encode_nnmp(X[nodes], D[:,:,iters+1,nodes], s, updatec_iter)
            
        for k in range(AtomN):
            multipliedMat = np.zeros([sigdim,sigdim,NodeN])
            index_temp = [[] for i in range(NodeN)]
            ER_temp = [[] for i in range(NodeN)]
            v_temp = [[] for i in range(NodeN)]
            
            for nodes in range(NodeN):
                indexes = np.nonzero(theta[nodes][k,:]!=0)[0]
                index_temp[nodes] = indexes
                
                if len(indexes)>0:
                    tempcoef = theta[nodes][:,indexes]
                    tempcoef[k,:] = 0
                    ER = X[nodes][:,indexes] - np.dot(D[:,:,iters+1,nodes], tempcoef)
                    ER_temp[nodes] = ER
                    multipliedMat[:,:,nodes] = np.dot(ER, ER.T)
                    
            if np.max(np.abs(multipliedMat))>0:
                newatom = distri_powermethod(vec_init, multipliedMat, NodeN, networkGraph, powerIterations, consensusIterations)
                flag = 0
                
                for nodes in range(NodeN):
                    if len(index_temp[nodes])>0:
                        v_temp[nodes] = np.dot(ER_temp[nodes].T, newatom[nodes,:])
                        if np.all(v_temp[nodes]<0)==True:
                            flag = 1
                            break
                        
                if flag==1:
                    newatom = -newatom
                    for nodes in range(NodeN):
                        if len(index_temp[nodes])>0:
                            v_temp[nodes] = np.dot(ER_temp[nodes].T, newatom[nodes,:])
                            
                newatom = (newatom>0)*newatom
                for nodes in range(NodeN):
                    if len(index_temp[nodes])>0:
                        v_temp[nodes] = (v_temp[nodes]>0)*v_temp[nodes]
                        
                for subiters in range(alteritern):
                    sumUpper = np.zeros([NodeN,sigdim])
                    sumLower = np.zeros(NodeN)
                    
                    for nodes in range(NodeN):
                        sumUpper[nodes,:] = np.dot(ER_temp[nodes], v_temp[nodes])
                        sumLower[nodes] = np.dot(v_temp[nodes], v_temp[nodes])
                        
                    for consiter in range(consensusIterations):
                        sumUpper = np.dot(networkGraph, sumUpper)
                        sumLower = np.dot(networkGraph, sumLower)
                        
                    division = np.tile(sumLower, (sigdim, 1)).T
                    newu = sumUpper/division
                    newu = (newu>0)*newu
                    
                    for nodes in range(NodeN):
                        if len(index_temp[nodes])>0:
                            v_temp[nodes] = np.dot(newu[nodes,:], ER_temp[nodes])/np.dot(newu[nodes,:], newu[nodes,:])
                            v_temp[nodes] = (v_temp[nodes]>0)*v_temp[nodes]
                            
                del newatom
                newatom = np.copy(newu)
                
                for nodes in range(NodeN):
                    if len(index_temp[nodes])>0:
                        normu = np.linalg.norm(newatom[nodes,:])
                        D[:,k,iters+1,nodes] = newatom[nodes,:]/normu
                        theta[nodes][k,index_temp[nodes]] = v_temp[nodes]*normu
                    
        dt = time.time() - t0
        print('the %dth iteration takes %f seconds' %(iters,dt))
        
    return D