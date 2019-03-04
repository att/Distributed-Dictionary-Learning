#Version
#--------
#Companion Code Version: 1.0
#
#
#Citation
#---------
#Any part of this code used in your work should be cited as follows:
#
#T. Wu, R. M. Rustamov and C. Goodall, "Distributed learning of human mobility patterns from cellular network data," 
#in Proc. 51st Annu. Conf. on Information Sciences and Systems (CISS), 2017, Companion Code, ver. 1.0.
#--------------------------------------------------------------------------
#
#CODE OUTLINE:
#We first generate synthetic data and then distribute data across sites in such a way that each site contains a subset 
#of the dictionary. Finally, representation errors for cloud K-SVD, localized K-SVD and centralized K-SVD are computed.


import time
import random
import numpy as np
from generategraph import gengraph
from sparsecoding import sparse_encode_omp
from centralizeddictionarylearning import dict_learning_ksvd
from distributeddictionarylearning import cloud_ksvd


def generate_dict(n_features, n_atom):
    D = np.random.randn(n_features, n_atom)
    D /= np.tile( np.sqrt((D ** 2).sum(axis=0)), (n_features, 1) )
    return D
    


NodeN = 10
p = 0.5
networkg = gengraph(NodeN, p)


""" Generate dictionary and data """
random.seed(time.time())
n_features = 20
TotalAtoms = 50
print("generating dictionary...")
D_gt = generate_dict(n_features, TotalAtoms)
LocalAtomN = 40
sp = 3
LocalTrainSamples = 150
TestSamples = 500
Y = [[] for i in range(NodeN)]    
    
for i in range(NodeN):
    indexes = np.random.permutation(range(TotalAtoms))[:LocalAtomN]
    tempD = D_gt[:,indexes]
    tempCoef = np.zeros([LocalAtomN, LocalTrainSamples])
    
    for j in range(LocalTrainSamples):
        atomidx = np.random.permutation(range(LocalAtomN))[:sp]
        tempCoef[atomidx,j] = np.random.randn(sp)
    
    tempY = np.dot(tempD, tempCoef)
    tempY /= np.tile( np.sqrt((tempY ** 2).sum(axis=0)), (n_features, 1) )
    Y[i] = tempY


Y_central = np.zeros([n_features, LocalTrainSamples*NodeN])
for i in range(NodeN):
    Y_central[:,np.arange(LocalTrainSamples*i,LocalTrainSamples*(i+1))] = Y[i]
    

TestCoef = np.zeros([TotalAtoms, TestSamples])
for i in range(TestSamples):
    atomidx = np.random.permutation(range(TotalAtoms))[:sp]
    TestCoef[atomidx,i] = np.random.randn(sp)
    
Ytest = np.dot(D_gt, TestCoef)
Ytest /= np.tile( np.sqrt((Ytest ** 2).sum(axis=0)), (n_features, 1) )

""" set parameters """
max_iter = 30
powerIterations = 5
consensusIterations = 10
D_init = generate_dict(n_features, TotalAtoms)
d_init = np.random.randn(1,n_features)
d_init /= np.linalg.norm(d_init)


""" distributed dictionary learning """
Dksvd = cloud_ksvd(Y, TotalAtoms, D_init, sp, NodeN, networkg, d_init, max_iter, powerIterations, consensusIterations)

TestDistriErrorMat = np.zeros([max_iter+1, NodeN])
for i in range(max_iter+1):
    print(i)
    for j in range(NodeN):
        temptheta2 = sparse_encode_omp(Ytest, Dksvd[:,:,i,j], sp)
        TestDistriErrorMat[i,j] = np.mean( ((Ytest - np.dot(Dksvd[:,:,i,j], temptheta2)) ** 2).sum(axis=0) )


""" localized dictionary learning """
D_local = np.zeros([n_features, TotalAtoms, max_iter+1, NodeN])
for i in range(NodeN):
    D_local[:,:,:,i] = dict_learning_ksvd(Y[i], D_init, sp, max_iter)

TestLocalErrorMat = np.zeros([max_iter+1, NodeN])
for i in range(max_iter+1):
    print(i)
    for j in range(NodeN):
        temptheta2 = sparse_encode_omp(Ytest, D_local[:,:,i,j], sp)
        TestLocalErrorMat[i,j] = np.mean( ((Ytest - np.dot(D_local[:,:,i,j], temptheta2)) ** 2).sum(axis=0) )


""" centralized dictionary learning """   
D_central = dict_learning_ksvd(Y_central, D_init, sp, max_iter)
TestCentralErrorMat = np.zeros(max_iter+1)
for i in range(max_iter+1):
    temptheta2 = sparse_encode_omp(Ytest, D_central[:,:,i], sp)
    TestCentralErrorMat[i] = np.mean( ((Ytest - np.dot(D_central[:,:,i], temptheta2)) ** 2).sum(axis=0) )