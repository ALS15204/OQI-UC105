import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import math
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

from scipy import linalg 

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


def DickeState(modes: int,photons: int,k:int=None) -> tuple:
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
	if(k!=None):
		dickebasis= [list(i)+[0]*k for i in dickebasis]

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

def order_of_magnitude(number):
	print("number", number)
	if number == 0:
		return -20
	print("number", number, "order_of_magnitude", math.floor(math.log(abs(number), 10)))
	return math.floor(math.log(abs(number), 10))

order_of_magnitude(10**(-6))

def get_unitary_la(A):
	P, D, V = linalg.svd(A)
	#print(np.allclose(A,P))
	m = len(A)
	#print(np.allclose(A, np.dot(P*D, V)))
	lsv = np.max(D)
	An = A/lsv

	#Added to debug
	'''
	An = np.dot(P*(D/lsv), V)
	U_An, D_An, V_An = linalg.svd(An)
	print("lsv of rescaled matrix", max(D_An))
	mat_norm = linalg.norm(An, ord=2) #spectral norm needs the ord=2 argument
	print("Matrix norm:", mat_norm)
	print("np.dot(An, An.conj().T)\n",np.dot(An, An.conj().T))
	print("\n",sqrtm(np.identity(m)-np.dot(An, An.conj().T)))'''
	#until here

	P = An
	if all([order_of_magnitude(value.real)<-9 for row in np.identity(m)-np.dot(An, An.conj().T) for value in row]):
		print("forcing Q = R = zeros")
		Q = R = np.zeros((m,m))
	else:
		print("Not unitary(?)")
		print(np.identity(m)-np.dot(An, An.conj().T))
		Q = sqrtm(np.identity(m)-np.dot(An, An.conj().T))
		R = sqrtm(np.identity(m)-np.dot(An.conj().T, An))
		print(Q)

	S = -An.conj().T
	Ubmat = np.bmat([[P, Q], [R, S]])
	Ubmat = Ubmat.real # WHY DOES IT NOT WORK WITHOUT THIS?
	return (np.copy(Ubmat), lsv)

def expectation_value_permanent(U,in_state,out_state=None):
	if(not out_state):
		out_state=in_state
	return (permanent(U,in_state, out_state=out_state))/np.sqrt(np.prod(factorial(in_state))*np.prod(factorial(out_state)))



class LimOh_sampler:
	"""_summary_
	This class contains a simplified implementation of theorem 2 by Lim & Oh. 
	As such are handled: input Fock states and output Fock states, input mode-separable states and output Fock states.
	The full implementation with mode separable input *and* output states is not currently implemented
	
	Ref:
	Lim, Y. & Oh, C. Efficient classical algorithms for linear optical circuits. 
	Preprint at https://doi.org/10.48550/arXiv.2502.12882 (2025).

	"""
	def __init__(self,input_state, output_state, unitary, nsamples,verbose=1):
		
		if(self.checkFock(input_state)):
			self.input_state = np.array([i[0][0] for i in input_state])
			self.output_state= output_state
			self.unitary = unitary
		elif(self.checkFock(output_state)):
			"Superposition detected in input, Fock in output. Reversed sampler"
			self.input_state = np.array([i[0][0] for i in output_state])
			self.output_state= input_state
			self.unitary = unitary.T.conj()
		else:
			"Case not covered. The approximator will be wrong :/"

		self.nsamples = nsamples
		self.trust=False
		
	def checkFock(self, state):
		return np.all(np.array([len(m)==1 for m in state]))
		

	def sampler(self,n, size=10000):
		"""
		Sample alpha from |P(alpha)|^2exp^{-|\alpha|^2} where P is the sum of projection of fock states onto coherent states
		
		As alpha can be complex, we simplify the input and restrict ourselves to pure Fock states ; so we stick to real positive alphas and sample the phase uniformly
		The overlap is then |<alpha|j>|^2 = alpha^{2j}/j! exp^{-|\alpha|^2} 
		We then use the gamma distribution x^(k-1)exp(-x)/Gamma(k)
		With  x=|alpha|^2 the pdf of the gamma distribution becomes # to |alpha|^{2k-1} exp^{-|\alpha|^2}
		We just need 2k-1 = j (the number of photon)
		
		We do not cover superposition states as it requires a complex weighted mixing of gamma distribution
		"""

		n = np.asarray(n)
		shape = (n + 1) / 2
		res = np.sqrt(np.random.gamma(shape=shape[..., None], scale=1, size=(*shape.shape, size)))

		return res * np.exp(1.j*np.random.uniform(0,2*np.pi,size=(*shape.shape, size)))

	def approximate(self):
		"""
		In this configuration, state_in is allways a Fock state
		"""
		alpha = self.sampler(self.input_state,self.nsamples)
		evolved_alpha = np.einsum('ij,jl->il',self.unitary, alpha)
		# print(self.input_state)

		# Computes alpha(modes,nsamples)**fockstate(modes) -> <alpha,psi> (modes, nsamples)
		# no need to conjugate
		denom = alpha ** self.input_state[:, None] / np.sqrt(factorial(self.input_state[:, None]))


		if(all(len(l) == len(self.output_state[0]) for l in self.output_state) and self.trust):
			num_photons_out = np.array([ np.array(a)[:,0] for a in self.output_state],dtype=int)
			amplitudes_out = np.array([ np.array(a)[:,1] for a in self.output_state],dtype=int)
			# print(amplitudes_out.shape)
			coherentproj = alpha[:, None, :] ** num_photons_out[:, :, None] /np.sqrt(factorial(num_photons_out[:, :, None]))
			# print(coherentproj.shape)
			num = np.einsum('ij,ijl->il',amplitudes_out.conj(),coherentproj)
		
		else:
			num_photons_out = [ np.array(a)[:,0] for a in self.output_state]#,dtype=int)
			amplitudes_out = [ np.array(a)[:,1] for a in self.output_state]#,dtype=int)

			# print(num_photons_out)
			# print(amplitudes_out)		
			# num photons can be of different size per mode, comprehension list instead of vectorized version
			num = np.array([
					amp_i.conj()@(alphai[None, :] ** ei[:, None] / np.sqrt(factorial(ei[:, None])))
					for alphai, ei, amp_i in zip(evolved_alpha, num_photons_out,amplitudes_out) ])
		
		x_alpha = np.prod(num/denom,axis=0)

		return np.abs(np.average(x_alpha))**2




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
