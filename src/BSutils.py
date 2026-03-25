import math 
import networkx as nx

import numpy as np
from math import comb, factorial
from itertools import combinations
import matplotlib.pyplot as plt


from typing import Union
from scipy.linalg import sqrtm


import perceval as pcvl
import exqalibur as xq

from scipy import linalg 

from scipy.special import factorial

from multiprocessing import Pool, cpu_count
from tqdm import tqdm



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

	return xq.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser")


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

		res+= xq.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser") #/len(input)
		checklist.append((i,xq.permanent_cx(pcvl.Matrix(submatrix), ptype="ryser"))) #/len(input)))

	return res,checklist

# ADD PERMANENT CALCULATION
def permanent(A, in_state, out_state=None):
	if(not out_state):
		out_state=in_state
	A = A.T
	modes= len(in_state)
	photons = sum(in_state)
	submatrix = np.concatenate([np.tile(A[i,:],in_state[i]).reshape((in_state[i],modes)) for i in np.where(in_state)[0] ])
	submatrix = np.concatenate([np.tile(submatrix[:,i],out_state[i]).reshape((out_state[i],photons)) for i in np.where(out_state)[0]])
	# print(submatrix)
	return xq.permanent_cx(submatrix, ptype="ryser")

def expectation_value_permanent(U,in_state,out_state=None):
	if(not out_state):
		out_state=in_state
	return (permanent(U,in_state, out_state=out_state))/np.sqrt(np.prod(factorial(in_state))*np.prod(factorial(out_state)))



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
	# print("number", number)
	if number == 0:
		return -20
	# print("number", number, "order_of_magnitude", math.floor(math.log(abs(number), 10)))
	return math.floor(math.log(abs(number), 10))

# order_of_magnitude(10**(-6))

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
		# print("forcing Q = R = zeros")
		Q = R = np.zeros((m,m))
	else:
		# print("Not unitary(?)")
		# print(np.identity(m)-np.dot(An, An.conj().T))
		Q = sqrtm(np.identity(m)-np.dot(An, An.conj().T))
		R = sqrtm(np.identity(m)-np.dot(An.conj().T, An))
		# print(Q)

	S = -An.conj().T
	Ubmat = np.bmat([[P, Q], [R, S]])
	Ubmat = Ubmat.real # WHY DOES IT NOT WORK WITHOUT THIS?
	return (np.copy(Ubmat), lsv)

def fockdim(m,n):
	return comb(m+n-1,n)

def CliffordSampler(U,state,Nsamples):
	
	backend = pcvl.Clifford2017Backend()
	backend.set_circuit(pcvl.Unitary(pcvl.Matrix(U)))
	backend.set_input_state(pcvl.BasicState(state))
	
	samples = backend.samples(Nsamples)
	return samples.count(pcvl.BasicState(state))

	



class LimOh_sampler:
	"""_summary_
	This class contains a simplified implementation of theorem 2 by Lim & Oh. 
	As such are handled: input Fock states and output Fock states, input mode-separable states and output Fock states.
	The full implementation with mode separable input *and* output states is not currently implemented
	
	Ref:
	Lim, Y. & Oh, C. Efficient classical algorithms for linear optical circuits. 
	Preprint at https://doi.org/10.48550/arXiv.2502.12882 (2025).

	"""
	def __init__(self,input_state, output_state, unitary, nsamples, verbose=1):
		
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
		self.trust=True
		self.disable=True

		self.result_values=[]
		self.result_var=[]
		self.chunk_sizes=[]

		if(verbose>=1):
			self.disable=False

	def checkFock(self, state):
		"""Check if state is a Fock state

		Args:
			state (_type_): _description_

		Returns:
			_type_: _description_
		"""
		return np.all(np.array([len(m)==1 for m in state]))
		

	def sampler(self,n, size=10000):
		"""
		Sample alpha from |P(alpha)|^2exp^{-|\alpha|^2} where P is the sum of projection of fock states onto coherent states
		
		As alpha can be complex, we simplify the input and restrict ourselves to pure Fock states ; so we stick to real positive alphas and sample the phase uniformly
		The overlap is then |<alpha|j>|^2 = alpha^{2j}/j! exp^{-|\alpha|^2} 
		We then use the gamma distribution x^(k-1)exp(-x)/Gamma(k)
		With  x=|alpha|^2 the pdf of the gamma distribution becomes # to |alpha|^{2k-1} exp^{-|\alpha|^2}
		We just need 2k-1 = j (the number of photon)
		
		We do not cover superposition states as it requires a complex weighted mixing of gamma distribution; 
		In the case of superposition states and Fock outputs, we compute the conjugate operation on Fock state as the sampling is less expensive for Fock state
		The superposition is left in output state for simplicity
		"""

		n = np.asarray(n)
		shape = (n + 1) / 2
		res = np.sqrt(np.random.gamma(shape=shape[..., None], scale=1, size=(*shape.shape, size)))

		return res * np.exp(1.j*np.random.uniform(0,2*np.pi,size=(*shape.shape, size)))

	def approximate(self, chunk_size=None, clear_cache=False):
		
		""" The function calls the approximator on different chunks using multiprocessing when enabled
			Each time the function is invoked, it stores in the class two lists (self.result_values,self.chunk_size) with estimator value and N_MC samples used correspongly.
			The final average is computed as the weighted sum of every call of this function (so getting more and more precise as we call it).

		Args:
			chunk_size: if not None, activates the multiprocessing with chunk sizes for each process ofn the pool
			clear_cache: cleans all the values stored for the cumulative MC estimation in (self.result_values,self.chunk_size)
		
		Returns:
			float: the approximator from theorem 2
		"""
		if(clear_cache):
				self.result_values=[]
				self.chunk_sizes=[]

		if(chunk_size==None):
			chunk_size=self.nsamples
			result = self.chunk_approximate(chunk_size)
			self.result_values.append(result)
			# self.result_variances.append(var)
			self.chunk_sizes.append(chunk_size)
		
		else:
			if(chunk_size<self.nsamples):
				chunk_size=self.nsamples
			"multiprocessing"
			pool = Pool(cpu_count())
			# results = pool.map(self.chunk_approximate, ((chunk_size) for _ in range(N)))
			N = self.nsamples//chunk_size
			for ans in tqdm(pool.imap(self.chunk_approximate, ((chunk_size) for _ in range(N))), total=N,disable=self.disable):
				self.result_values.append(ans)
				self.chunk_sizes.append(chunk_size)

		res = np.average(self.result_values,weights=self.chunk_sizes)
		return np.abs(res)**2
	
	def get_current_value(self):
		res = np.average(self.result_values,weights=self.chunk_sizes)
		return np.abs(res)**2

	def chunk_approximate(self,chunksize,write=False):
		"""
		In this configuration, state_in is allways a Fock state
		"""
		# alpha is of shape (modes, Nsamples)
		alpha = self.sampler(self.input_state,chunksize)
		# Computing | U alpha> for each alpha
		evolved_alpha = np.einsum('ij,jl->il',self.unitary, alpha)
		

		# Computes alpha(modes,nsamples)**fockstate(modes) -> <alpha,psi> (modes, nsamples)
		# no need to conjugate (formula already adapted)
		denom = alpha ** self.input_state[:, None] / np.sqrt(factorial(self.input_state[:, None]))

		# The following is trying to accelerate things when the output state (possibly in superposition can be vectorized
		if(all(len(l) == len(self.output_state[0]) for l in self.output_state) and self.trust):
			num_photons_out = np.array([ np.array(a)[:,0] for a in self.output_state],dtype=int)
			amplitudes_out = np.array([ np.array(a)[:,1] for a in self.output_state],dtype=complex)
			# print(amplitudes_out.shape)

			# Computes U alpha(modes,nsamples)**fockstate(modes) ->  <phi|U alpha> (modes, nsamples)
			coherentproj = evolved_alpha[:, None, :] ** num_photons_out[:, :, None] /np.sqrt(factorial(num_photons_out[:, :, None]))
			
			num = np.einsum('ij,ijl->il',amplitudes_out.conj(),coherentproj)
		
		else:
			# non vectorised version, list comprehension instead
			num_photons_out = [ np.array(a)[:,0] for a in self.output_state]
			amplitudes_out = [ np.array(a)[:,1] for a in self.output_state]

			
			# num photons can be of different size per mode, comprehension list instead of vectorized version
			# Computes U alpha(modes,nsamples)**fockstate(modes) -> <phi|U alpha> (modes, nsamples)
			num = np.array([
					amp_i.conj()@(alphai[None, :] ** ei[:, None] / np.sqrt(factorial(ei[:, None])))
					for alphai, ei, amp_i in zip(evolved_alpha, num_photons_out,amplitudes_out) ])
		
		# product of each mode
		x_alpha = np.prod(num/denom,axis=0)
		# average of all alphas
		mean = np.average(x_alpha)
		# variance = np.average((x_alpha-mean)**2)

		return mean



class find_densest_subgraph():
	def __init__(self,G, k, method='exact', Nsamples=100_000, verbose=0):
		"""_summary_

		Args:
			G (graph): adjacency matrix of the graph
			k (int, optional): size of the subgraph to search for
		"""

		m = G.number_of_nodes()
		adj = nx.adjacency_matrix(G).toarray()

		# if(method='exact'):

		# if(superposition):
		output = [ 
				[1 if i in c else 0 for i in range(m)]
				for c in combinations(np.arange(m), k )
				if(nx.is_connected(G.subgraph(c)))
			]
		output_extended = [ 
				[1 if i in c else 0 for i in range(m)]+[0]*m
				for c in combinations(np.arange(m), k )
				if(nx.is_connected(G.subgraph(c)))
			]
		output_translated = [
				[ [(i,1)]  for i in o ] +[[(0,1)]]*m
					for o in output
				]
		
		keys = [''.join(  [str(e) for e in l] ) for l in output]
		
		if(method=="exact"):
			self.res_pure = { k: np.abs(perm_estimation_graph(adj,o)) for k,o in zip(keys,output) }

		if(method=="sampler"):
			unitary,lsv = get_unitary_la(adj)
			self.res_pure = { k: np.abs(CliffordSampler(unitary,o,Nsamples)) for k,o in zip(keys,output_extended) }


		if(method=="approximate"):
			unitary,lsv = get_unitary_la(adj)
			state_translator = [ [(0,1),(1,1)] ]*m
			s_in = state_translator +[[(0,1)]]*m
			# print(s_in)
			# print(output_translated[0])
			MCsamplers = [LimOh_sampler(s_in,s_o,unitary,Nsamples,verbose=verbose) for s_o in output_translated]

			self.MCestimators = dict(zip(keys,MCsamplers))
			self.run()

		# self.MCestimator = LimOh_sampler(s_in,s_o,unitary,1_000_000,verbose=verbose)

	def results_pure(self):
		res = list(self.res_pure.items())
		res.sort(key=lambda item: item[1], reverse=True)
		return res

	def run(self,chunk_size=100_000):
		for v in self.MCestimators.values():
			v.approximate(chunk_size=chunk_size)

	
	def results_estimators(self):
		res = list(self.MCestimators.items())
		res.sort(key=lambda item: item[1].get_current_value(), reverse=True)
		return res
	
	

def is_subclique(G,nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n*(n-1)/2


def find_maxclique(G,method='exact',out_candidates=1,verbose=0,Nsamples=100_000):
	"""_summary_

	Args:
		G (_type_): nx graph
		method (str, optional): 'exact', 'approximate' for the permanent calculation. 
								-'exact': by default using Ryser, and Fock input state
								-'approximate': uses Lim & Oh algo for estimating SBS expectation values
								-'sampler': uses perceval sampler for Fock input state
		out_candidates (int, optional): number of candidates expected. Defaults to 1.

	Returns:
		list: list of cliques candidates
	"""
	counter = 0
	cliques = []
	m = G.number_of_nodes()
	candidates_k = []
	for k in range(m,2,-1):
		if(verbose>=1):
			print("search for ", k)

		finder = find_densest_subgraph(G,k,method=method,Nsamples=Nsamples,verbose=verbose-1)

		if(method=='exact' or method=='sampler'):
			l = finder.results_pure()[:20]
			l =  [x[0] for x in l if x[1] == l[0][1]]
			
			for candidate in l:
				nodes = [i for i,e in enumerate(list(candidate)) if e=='1' ]
				candidates_k.append(nodes)

				if(is_subclique(G,nodes) and (counter < out_candidates)):
					counter+=1
					# print(nodes)
					cliques.append(nodes)
				if(counter==out_candidates):
					return cliques
		
		else:
			#should be run many times for accurate estimators...
			finder.run()

			l = finder.results_estimators()[0]
			nodes = [i for i,e in enumerate(l) if e=='1' ]
			cliques.append(nodes)
	return cliques


		# maxclique = np.argmax(self.MCestimators.values())
		# return self.MCestimators.keys()[maxclique], self.MCestimators.values()[maxclique].get_current_values()

# Dstate


# The following "torch" version contains some bugs

# import torch
# import math

# class LimOh_sampler_torch:
# 	"""
# 	PyTorch version of Lim & Oh sampler (Theorem 2 simplified)
# 	Handles:
# 	  - Fock input -> Fock output
# 	  - Mode-separable input -> Fock output (partial)
# 	"""

# 	def __init__(self, input_state, output_state, unitary, nsamples, device="cpu", verbose=1):
# 		self.device = device
# 		self.verbose = verbose
# 		self.nsamples = nsamples
# 		self.trust = False  # advanced mode

# 		# Convert unitary to torch complex tensor
# 		unitary = torch.as_tensor(unitary, dtype=torch.cfloat, device=device)

# 		if self.checkFock(input_state):
# 			self.input_state = torch.tensor(
# 				[mode[0][0] for mode in input_state],
# 				dtype=torch.long,
# 				device=device
# 			)
# 			self.output_state = output_state
# 			self.unitary = unitary

# 		elif self.checkFock(output_state):
# 			if verbose:
# 				print("Superposition detected in input, Fock in output. Reversed sampler")
# 			self.input_state = torch.tensor(
# 				[mode[0][0] for mode in output_state],
# 				dtype=torch.long,
# 				device=device
# 			)
# 			self.output_state = input_state
# 			self.unitary = unitary.T.conj()

# 		else:
# 			raise ValueError("Case not covered: both input and output are superpositions.")

# 	@staticmethod
# 	def checkFock(state):
# 		return all(len(mode) == 1 for mode in state)

# 	def sampler(self, n, size):
# 		"""
# 		Sample coherent amplitudes alpha
# 		Output shape: (modes, size)
# 		"""
# 		n = torch.as_tensor(n, dtype=torch.float32, device=self.device)
# 		shape = (n + 1) / 2.0                       # (modes,)

# 		gamma = torch.distributions.Gamma(
# 			shape,
# 			torch.ones_like(shape)
# 		)

# 		# |alpha|^2 : (modes, size)
# 		r2 = gamma.sample((size,)).T                # (modes, size)
# 		r = torch.sqrt(r2)

# 		phase = 2 * math.pi * torch.rand_like(r)
# 		alpha = r * torch.exp(1j * phase)

# 		return alpha


# 	def approximate(self, chunk_size=100_000):
# 		"""
# 		Monte-Carlo approximation with chunking to handle very large nsamples
# 		chunk_size: number of samples to process at once to avoid matmul errors
# 		"""
# 		modes = len(self.input_state)
# 		total_samples = self.nsamples
# 		result = 0.0
# 		processed = 0

# 		while processed < total_samples:
# 			this_chunk = min(chunk_size, total_samples - processed)
# 			alpha = self.sampler(self.input_state, this_chunk)  # (modes, this_chunk)
# 			alpha = alpha.to(dtype=torch.cfloat, device=self.device)

# 			# Linear optical evolution
# 			evolved_alpha = torch.mm(self.unitary, alpha)  # (modes, this_chunk)

# 			# Denominator <alpha|psi_in>
# 			denom = alpha ** self.input_state[:, None] / torch.exp(
#     0.5 * torch.special.gammaln(self.input_state[:, None] + 1)
# )
			
# 			# Numerator <psi_out|alpha'>
# 			if self.trust and all(len(l) == len(self.output_state[0]) for l in self.output_state):
# 				num_photons_out = torch.tensor([torch.tensor(a)[:, 0] for a in self.output_state],
# 											dtype=torch.long, device=self.device)
# 				amplitudes_out = torch.tensor([torch.tensor(a)[:, 1] for a in self.output_state],
# 											dtype=torch.cfloat, device=self.device)
# 				coherent_proj = evolved_alpha[:, None, :] ** num_photons_out[:, :, None] / torch.exp(
#     0.5 * torch.special.gammaln(num_photons_out[:, :, None] + 1)
# )

# 				num = torch.einsum("ij,ijl->il", amplitudes_out.conj(), coherent_proj)

# 			else:
# 				num_list = []
# 				for alphai, state_i in zip(evolved_alpha, self.output_state):
# 					ei = torch.tensor(state_i)[:, 0].to(self.device)
# 					amp_i = amp_i = torch.tensor(state_i, device=self.device, dtype=torch.cfloat)[:, 1]
# 					proj = alphai[None, :] ** ei[:, None] / torch.exp(
#     0.5 * torch.special.gammaln(ei[:, None] + 1)
# )
# 				num_list.append(amp_i.conj() @ proj)
# 				num = torch.stack(num_list, dim=0)

# 			x_alpha = torch.prod(num / denom, dim=0)
# 			result += x_alpha.sum()  # accumulate sum
# 			processed += this_chunk

# 		return torch.abs(result / total_samples) ** 2


