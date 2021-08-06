import numpy as np
import random
import networkx as nx


def unbiased_controlled_rounding(A :np.array, tol=0.0001) -> np.array: #A: two-way table comprising non-negative real-number entries
    ###### Unibased controlled rounding
    ## Ref: Cox, Lawrence H. "A constructive procedure for unbiased controlled rounding." 
    ## Journal of the American Statistical Association 82.398 (1987): 520-524.

    m=A.shape[0]
    n=A.shape[1]

    colsum = np.sum(A, axis=0)
    new_col = (1- np.mod(colsum, 1)).reshape(1,n)
    C=np.concatenate((A,new_col),axis=0)

    rowsum = np.sum(C, axis=1)
    new_row = 1- np.mod(rowsum, 1).reshape(m+1,1)
    C=np.concatenate((C,new_row),axis=1)
    
    while( ( ((np.mod(C, 1) > tol) & (np.mod(C, 1) < (1-tol)) )  *1).sum() > 0):
        edges=[]
        fracs=np.random.permutation(np.argwhere(((np.mod(C, 1) > tol) & (np.mod(C, 1) < (1-tol)) )))
        for (x,y) in fracs:
            y=y+m+1
            edges.append((x,y))

        G = nx.DiGraph(edges)
        cycle= nx.find_cycle(G, orientation="ignore")
        path_row=0
        if cycle[0][0]==cycle[1][0]: path_row=1

        L=[]
        for (x,y,direct) in cycle:
            L.append([x,y-m-1])

        d_minus = 1
        d_plus = 1
        for idx, [i,j] in enumerate(L):
            if idx % 2 == 0:
                d_minus= min(np.mod(C[i][j], 1),d_minus)
                d_plus= min((1-np.mod(C[i][j], 1)),d_plus)
            else:
                d_minus= min((1-np.mod(C[i][j], 1)),d_minus)
                d_plus= min(np.mod(C[i][j], 1),d_plus)

        p_minus=d_plus/(d_plus+d_minus)
        p_plus=d_minus/(d_plus+d_minus)

        select_minus= np.random.choice([1,0], 1, p=[p_minus,p_plus] )
        if select_minus:
            for idx, [i,j] in enumerate(L):
                if idx % 2 == 0:
                    C[i][j]-=d_minus
                else:
                    C[i][j]+=d_minus
        else:
            for idx, [i,j] in enumerate(L):
                if idx % 2 == 0:
                    C[i][j]+=d_plus
                else:
                    C[i][j]-=d_plus 

    C= np.round(C)
    
    return C[0:m,0:n]

# # Example

A= np.array([[0.91,0.7,1.23,0.1],[1.21,2,2.83,0.2],[1.21,0.2,1.03,0.5]])

print(A)
print('\n')
print(unbiased_controlled_rounding(A))

m=A.shape[0]
n=A.shape[1]
T=1000
simulation=np.zeros((T,m,n))

for i in range(T):
    simulation[i]= unbiased_controlled_rounding(A)

mean_simulation=np.average(simulation,axis=0)
print(mean_simulation)

print(A-mean_simulation)