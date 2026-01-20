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

import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import math
from math import comb, factorial
from itertools import combinations
from functools import partial

import matplotlib.pyplot as plt


from typing import Union

from tabulate import tabulate
import pandas as pd

from scipy.linalg import sqrtm,eig


import perceval as pcvl
import exqalibur as xqlbr

import perceval as pcvl

from BSutils import *
from scipy.stats import unitary_group

from itertools import permutations
from scipy.special import factorial
import vegas



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


def superposition_perm(adj,out_state:list, k:int=None):
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


def superposition_perm_bis(adj,input:list, out_state:list):
    # out_nodes = [index for index, num in enumerate(list(out_state)) if num == 1]
    n = len(adj)
    
    # input = np.array(
    # [
    #     [1 if i in comb else 0 for i in range(n)]
    #     for comb in combinations(np.arange(n), k)
    # ]
    # )

    res = 0
    checklist=[]

    for i in input:
        if (sum(i)!=sum(out_state)):
            print("lol")
            continue
        submatrix = adj[:,np.where(i)[0]]
        submatrix = submatrix[np.where(out_state)[0],:]

        res+= xqlbr.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser") #/len(input)
        checklist.append((i,xqlbr.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser"))) #/len(input)))

    return res,checklist


# ADD PERMANENT CALCULATION
def permanent(A, in_state, out_state=None):
	if(not out_state):
		out_state=in_state
	# print(A)
	# submatrix = A[:,np.where(in_state)[0]]
	# submatrix = submatrix[np.where(out_state)[0],:]
	# print(submatrix)
	# print(A)
	
	A = A.T
	modes= len(in_state)
	photons = sum(in_state)
	submatrix = np.concatenate([np.tile(A[i,:],in_state[i]).reshape((in_state[i],modes)) for i in np.where(in_state)[0] ])
	submatrix = np.concatenate([np.tile(submatrix[:,i],out_state[i]).reshape((out_state[i],photons)) for i in np.where(out_state)[0]])
	# print(submatrix)
	return xqlbr.permanent_cx(submatrix, ptype="ryser")


def DickeState(modes: int,photons: int) -> tuple:
    """Return a Dicke state with given number of modes & photons
        A Dicke state D(m,n)) is defined as the coherent superposition of Fock states with n unbunched photons within m modes.
        (he superposition if composd of n choose m Fock states)
    Args:
        modes (int): number of modes
        photons (int): number of photons

    Returns:
       (pcvl.StateVector, list): Dicke state along with the corresponding list of Focj states
    """
    if(photons > modes):
        print("Fallback to case photons = modes")
        photons=modes

    dickebasis = list(set(permutations([1]*photons+[0]*(modes-photons))))
    dicke_state_pcvl=pcvl.StateVector(dickebasis[0])
    for d in dickebasis[1:]:
        dicke_state_pcvl+= pcvl.StateVector(d)
    return dicke_state_pcvl,dickebasis



def compute_simbackend_amplitudes(unitary, instate, outstate):
        # Fastest
        U = pcvl.Unitary(pcvl.Matrix(unitary))
        simulator = pcvl.Simulator(pcvl.SLOSBackend())
        simulator.set_circuit(U)

        res = 0
        for el in outstate:
            ramp = 0
            for i,coeff in instate:
                ramp += simulator.prob_amplitude(i,el)*coeff
            res += np.abs(ramp)**2
        
        return res



def expectation_value_permanent(U,in_state,out_state=None):
	if(not out_state):
		out_state=in_state
	return (permanent(U,in_state, out_state=out_state))/np.sqrt(np.prod(factorial(in_state))*np.prod(factorial(out_state)))


def coherent_state(alpha,cherrypick=None,photon_cut=2):
	alpha = np.asarray(alpha)
	if not cherrypick:
		cherrypick=range(photon_cut)
	# if(cherrypick):
	# 	return np.array(zip(cherrypick, )))
	return np.array([np.exp(-np.abs(alpha)**2/2)*alpha**n /np.sqrt(factorial(n)) 
		    for n in cherrypick], dtype=complex).T.swapaxes(0,1)
	#shape (alpha.shape[0], mode, photon_cut/cherrypick)

# def state_repr(state,photon_cut=2):
# 	modes = len(state)
# 	res =  np.zeros((modes,photon_cut))
# 	for i in range(modes):
# 		res[i,state[i]] = 1
# 	#shape (mode, photon_cut)
# 	return res



# def q_distr(alpha, state_in, cherrypick=False,photon_cut=2):
# 	alpha = np.asarray(alpha)
# 	if(len(alpha.shape)==0):
# 		alpha = alpha.reshape(1,1)
# 	elif(len(alpha.shape)==1):
# 		alpha=alpha.reshape(alpha.shape[0],1)
# 	modes = len(state_in)
# 	if not cherrypick:
# 	# Full calculation
# 		coherent_alpha = coherent_state(alpha,photon_cut=photon_cut)
# 		# print( (np.abs(np.tensordot(coherent_alpha.conj(), state_in, axes=([2,1])))**2).shape)
# 		M = np.abs(np.tensordot(coherent_alpha.conj(), state_in, axes=([2,1])))**2
# 		q = np.prod(np.diagonal(M, axis1=1, axis2=2), axis=1)
# 		return 1/(np.pi**modes)* q
# 	else:
# 		# state_in array [modes,[(num_photon,amplitude)]]
# 		q_cumul = 1
# 		for i in range(modes):
# 			num_photons_pick,amplitudes = zip(*state_in[i])
# 			coherent_alpha = coherent_state(alpha[:,i],cherrypick=num_photons_pick).T
# 			# print("coherent", coherent_alpha)
# 			# print("amplitudes",amplitudes)
# 			q = np.abs(coherent_alpha.conj()@np.array(amplitudes))**2
# 			q_cumul*=q
# 		return 1/(np.pi**modes)* q_cumul



def q_pdf(alpha, state_in):
	# modes = np.asarray(state_in).shape[0]
	# q_modes = np.ones(modes)
	alpha = np.asarray(alpha)
	if(len(alpha.shape)==0):
		alpha = alpha.reshape(1,1)
	elif(len(alpha.shape)==1):
		alpha=alpha.reshape(alpha.shape[0],1)
		
	q_modes=[]
	for i,s in enumerate(state_in):
		num_photons = np.array([ a[0] for a in s],dtype=int)
		amplitudes = np.array([ a[1] for a in s])

		coherent_factor = np.tile(np.exp(-np.abs(alpha[:,i])**2/2),len(num_photons)).reshape(len(num_photons),alpha.shape[0]).T \
								* (np.tile(alpha[:,i],len(num_photons)).reshape(len(num_photons),alpha.shape[0]).T **num_photons )/np.sqrt(factorial(num_photons))
		coherent_factor=coherent_factor.conj()

		scalarprod =  coherent_factor @ amplitudes
		
		q_modes.append(np.abs(scalarprod)**2/np.pi )
	# np.exp(-np.abs(alpha)**2/2)/np.pi * sum(
	return np.prod(q_modes,axis=0)




def X_pdf(alpha,state_in,state_out,unitary):
	alpha = np.asarray(alpha)
	if(len(alpha.shape)==0):
		alpha = alpha.reshape(1,1)
	elif(len(alpha.shape)==1):
		alpha=alpha.reshape(alpha.shape[0],1)

	evolved_alpha = np.einsum('ij,lj->li', unitary, alpha)

	q_modes=[]
	for i,(s,so) in enumerate(list(zip(state_in,state_out))):
		num_photons = np.array([ a[0] for a in s],dtype=int)
		amplitudes = np.array([ a[1] for a in s])

		num_photons_out = np.array([ a[0] for a in so],dtype=int)
		amplitudes_out = np.array([ a[1] for a in so])

		coherent_factor = np.tile(np.exp(-np.abs(alpha[:,i])**2/2),len(num_photons)).reshape(len(num_photons),alpha.shape[0]).T \
								* (np.tile(alpha[:,i],len(num_photons)).reshape(len(num_photons),alpha.shape[0]).T **num_photons )/np.sqrt(factorial(num_photons))
		coherent_factor=coherent_factor.conj()

		evolved_coherent_factor = np.tile(np.exp(-np.abs(evolved_alpha[:,i])**2/2),len(num_photons_out)).reshape(len(num_photons_out),evolved_alpha.shape[0]).T \
								* (np.tile(evolved_alpha[:,i],len(num_photons_out)).reshape(len(num_photons_out),evolved_alpha.shape[0]).T **num_photons_out )/np.sqrt(factorial(num_photons_out))
		
		num = evolved_coherent_factor @ amplitudes_out.conj()
		# print(coherent_factor.shape,amplitudes.shape)
	
		denom = (coherent_factor @ amplitudes).conj()
		# print(num.shape,denom.shape)
		q_modes.append(num/denom)
	# 	print("lol")
	# 	print(num,denom)
	# 	print("lol2", (num/denom))
	# print("lol3",q_modes)
	return np.prod(q_modes,axis=0)

def sample_qmode(state_in,size=1000, method='cumulative'):
	if(method=='cumulative'):	
		x = np.linspace(0,5,100*size+1)
		jac = np.abs(x)
		cumulative = np.cumsum(q_pdf(x.reshape((100*size+1,1)),state_in)*jac*2*np.pi)/(100*size)
		a = np.random.uniform(0,cumulative[-1],size)
		# print(cumulative[-1])
		res = np.searchsorted(cumulative,a )
		return x[res],np.random.uniform(0,2*np.pi,size=size)
		

def approx(state_in,state_out,unitary,size=1000):
	cumul=1
	alphas = np.zeros((size,len(state_in)),dtype=complex)

	for i,s in enumerate(state_in):
		s = np.array([s])
		# samples_alpha,wgts = sample_qmode(sv,size=size)
		samples_alpha,phases = sample_qmode(s,size=size)
		alphas[:,i] = samples_alpha*np.exp(1.j*phases)

	
	return np.average(X_pdf(alphas,state_in,state_out,unitary))
