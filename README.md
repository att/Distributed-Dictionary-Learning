# Distributed Dictionary Learning

Distributed Dictionary Learning is a set of python modules for learning a dictionary when
the data and computational resources are distributed across multiple nodes in a network.

The basic premise is that natural signals can be well approximated using sparse linear combinations
of a few vectors, also called atoms, in some overcomplete basis, or so-called dictionary.

We consider the case when it is cost- and time-prohibitive to gather the distributed data
to a central location.   We then learn a collection of dictionaries through local computations
together within inter-nodal computation to obtain a collaborative dictionary.

The algorithm is cloud NN-K-SVD, meaning cloud = distributed, NN for non-negative weights
given each of the atoms for each observation, K for dimension of the overcomplete basis, and
SVD for the underlying singular value decomposition approach applied iteratively.

Simpler and less effective variants are local K-SVD, local NN-K-SVD, K-SVD, and NN-K-SVD.

For an application to anonymized call data see [CISS 2017 paper] and [CISS 2017 talk].
The paper is also part of the [IEEE Digital Library].

[CISS 2017 paper]: https://www.github.com/att/distributed-dictionary-learning/CISS2017_Proceedings.pdf
[CISS 2017 talk]: https://www.github.com/att/distributed-dictionary-learning/CISS2017_Talk.pdf
[IEEE Digital Library]: https://ieeexplore.ieee.org/document/7926085
[paper]: https://www.github.com/att/distributed-dictionary-learning/CISS2017_Proceedings.pdf
[talk]: https://www.github.com/att/distributed-dictionary-learning/CISS2017_Talk.pdf

Developed for AT&T by Tong Wu (https://github.com/twugithub)
and by Colin Goodall (https://github.com/ColinGoodall) and Raif Rustamov

## Alternative approaches

The two synthetic data Python source files provide programs for distributed dictionary learning,
for K-SVD and NN-K-SVD.   For the local alternative, run as follows:-

To fit an overcomplete approximating basis, run as follows:-

Typically, each calculation involves iteration of 2 parts, as shown in the [paper] and [talk].

* Part 1 is sparse coding, where each site computes locally the sparse codes for its local training data.

* Part 2 is the dictionary update, collaboratively across the sites using inter-site connections specified in the network graph.

## Overview of Python Source Files

Each file is 30 - 140 lines of python

* cloudksvd_syntheticdata_main.py

companion code to CISS 2017 paper; start here for cloud K-SVD

* cloudnnksvd_syntheticdata_main.

start here for cloud NN-K-SVD

* generategraph.py

generate random network graph, function gengraph, used with synthetic data routines

* distributeddictionarylearning.py

defines fucntions cloud_ksvd and cloud_nnksvd
includes in arguments the network graph

* dictionaryupdate.py

functions 12_update_dict, update the dictionary for K-SVD
nn12_update_dict, update the dictionary for NN-KSVD

* centralizeddictionarylearning.py

functions dict_learning_ksvd, dict_learning_nnksvd

* distributedpowermethod.py

function distri_powermethod for power method across network graph

* sparsecoding.py

functions omp and sparse_encode_omp for orthogonal matching pursuit, and
nnmp and sparse_encode_nnmp for nonnegative matching pursuit

* breathfirstsearch.py

function bfs to determine paths through the network from the
adjacency matrix of the network, the starting node, and the destination node;
function find_paths given the spanning tree output from bfs

