# Distributed Dictionary Learning

Distributed Dictionary Learning is a set of python modules for learning a dictionary when
the data and computational resources are distributed across multiple nodes in a network.

The basic premise is that natural signals can be well approximated using sparse linear combinations
of a few vectors, also called atoms, in some overcomplete basis, or so-called dictionary.

We consider the case when it is cost- and time-prohibitive to gather the distributed data
to a central location. We then learn a collection of dictionaries through local computations
together within inter-nodal computation to obtain a collaborative dictionary.

The algorithm is cloud NN-K-SVD, meaning cloud = distributed, NN for non-negative weights
given each of the atoms for each observation, K for dimension of the overcomplete basis, and
SVD for the underlying singular value decomposition approach applied iteratively.

Simpler and less effective variants are local K-SVD, local NN-K-SVD, K-SVD, and NN-K-SVD.

For an application to anonymized call data see [paper] and [talk].
The paper is also part of the [IEEE Digital Library].

[IEEE Digital Library]: https://ieeexplore.ieee.org/document/7926085
[paper]: https://github.com/att/Distributed-Dictionary-Learning/blob/master/CISS2017_Proceedings.pdf
[talk]: https://github.com/att/Distributed-Dictionary-Learning/blob/master/CISS2017_Talk.pdf

Developed for AT&T by Tong Wu (https://github.com/twugithub), Colin Goodall (https://github.com/ColinGoodall) and Raif Rustamov

## Citation
Any part of this code used in your work should be cited as follows:

T. Wu, R. M. Rustamov and C. Goodall, "Distributed learning of human mobility patterns from cellular network data," in Proc. 51st Annu. Conf. on Information Sciences and Systems, 2017. Companion Code, ver. 1.0.

## Reporting of issues
Any issues in this code should be reported to T. Wu. However, this companion code is being provided on an "As IS" basis to support the ideals of reproducible research. As such, no guarantees are being made that the reported issues will be eventually fixed.

## Computational environment
This code has been tested in Windows 7, 10/Linux with Python 3.7. While it can run in other environments also, we can neither provide such guarantees nor can help you make it compatible in other environments.

## Overview of Python source files

The two synthetic data Python source files provide programs for distributed dictionary learning based on K-SVD and NN-K-SVD.

Typically, each calculation involves iteration of 2 parts, as shown in the [paper] and [talk].

* Part 1 is sparse coding, where each site computes locally the sparse codes for its local training data.

* Part 2 is the dictionary update, collaboratively across the sites using inter-site connections specified in the network graph.

The detailed instructions are as follows:

* cloudksvd_syntheticdata_main.py and cloudnnksvd_syntheticdata_main.py - examples of the usage of the distributed/centralized algorithms using synthetic data

* generategraph.py - generates a random network graph

* distributeddictionarylearning.py - the implementation of cloud K-SVD and cloud NN-K-SVD

* dictionaryupdate.py - function 12_update_dict for updating the dictionary in K-SVD; nn12_update_dict for updating the dictionary in NN-K-SVD

* centralizeddictionarylearning.py - the implementation of K-SVD and NN-K-SVD

* distributedpowermethod.py - distributed power method for the generated network graph

* sparsecoding.py - functions omp and sparse_encode_omp for orthogonal matching pursuit, and nnmp and sparse_encode_nnmp for nonnegative matching pursuit

* breathfirstsearch.py - function bfs determines paths through the network from the adjacency matrix of the network, the starting node, and the destination node; function find_paths gives the spanning tree output from bfs

