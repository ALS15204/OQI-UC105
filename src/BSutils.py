import networkx as nx

import numpy as np
import matplotlib.pyplot as plt 
from math import comb, factorial
from itertools import combinations
from functools import partial

from typing import Union

from tabulate import tabulate
import pandas as pd

from scipy.linalg import sqrtm,eig


import perceval as pcvl
import exqalibur as xqlbr


def density(G,state):
    node_list = [index for index, num in enumerate(list(state)) if num == 1]
    return nx.density(G.subgraph(node_list))


def perm_estimation_graph(G: Union[nx.Graph,np.array], state):
    
    node_list = [index for index, num in enumerate(list(state)) if num == 1]
    if(type(G)==nx.Graph):
        submatrix = nx.to_numpy_array(G.subgraph(node_list))
    else:  
        submatrix = G[:,np.where(state)[0]]
        submatrix = submatrix[np.where(state)[0],:]

    return xqlbr.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser")


def superposition_perm(adj,out_state, k=None):
    # out_nodes = [index for index, num in enumerate(list(out_state)) if num == 1]
    n = len(adj)
    if(k==None):
        k = sum(np.array(out_state))

    input = np.array(
    [
        [1 if i in comb else 0 for i in range(n)]
        for comb in combinations(np.arange(n), k)
    ]
    )
    res = 0
    checklist=[]

    for i in input:
        submatrix = adj[:,np.where(i)[0]]
        submatrix = submatrix[np.where(out_state)[0],:]

        res+= xqlbr.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser") #/len(input)
        checklist.append((i,xqlbr.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser"))) #/len(input)))

    return res,checklist
