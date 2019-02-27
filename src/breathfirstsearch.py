import numpy as np


def find_paths(sp_tree,u,v):
    if sp_tree[v,u] == 1:
        path = [[v]]
    else:
        neighbors = list(np.where(sp_tree[v,:]==1)[0])
        path = []
        for i in range(len(neighbors)):
            pp = find_paths(sp_tree,u,neighbors[i])
            for j in range(len(pp)):
                pp[j].append(v)
            
            path = path + pp
            del pp
    
    return path
    
    
def bfs(A,s,t):
    
    """
    Parameters
    ----------
    Inputs:
        A: adjacency matrix of the network
        s: starting node
        t: destination node
    Outputs:
        sp_tree: spanning tree created from node s
        distance: a vector containing distances from s to all other nodes
        found: a boolean value indicates if there is a link from s to t
        shortest_paths: all the shortest paths from s to t
    """
    
    distance = np.ones([np.shape(A)[0]])*-1
    distance[s] = 0
    queue = np.zeros([np.shape(A)[0]]).astype(int)
    queue[0] = s
    read_pointer = 0
    write_pointer = 1
    sp_tree = np.zeros([np.shape(A)[0],np.shape(A)[0]])
    found = 0

    while read_pointer != write_pointer:
        d = distance[queue[read_pointer]]
        neighbor = list(set(list(np.where(A[queue[read_pointer],:]==1)[0])) - set(list([queue[read_pointer]])))
        for i in range(len(neighbor)):
            if distance[neighbor[i]] == -1:
                distance[neighbor[i]] = d + 1
                queue[write_pointer] = neighbor[i]
                write_pointer = write_pointer + 1
                sp_tree[neighbor[i],queue[read_pointer]] = 1
            elif distance[neighbor[i]] == d+1:
                sp_tree[neighbor[i],queue[read_pointer]] = 1
                
            if neighbor[i] == t:
                found = 1
                
        read_pointer = read_pointer + 1
        
    if found == 1:
        if s == t:
            shortest_paths = []
        else:
            shortest_paths = find_paths(sp_tree,s,t)
    else:
        shortest_paths = []
        
    return sp_tree, distance, found, shortest_paths