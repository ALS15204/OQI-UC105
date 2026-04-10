import networkx as nx
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
import scipy.io
import os
import pickle

def load_graph(path):
    """Load graph from DIMACS, Matrix Market, or Pickle format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
        
    name = os.path.splitext(os.path.basename(path))[0]
    
    if path.endswith('.mtx'):
        mat = scipy.io.mmread(path).tocsr()
        mat = mat + mat.T # Ensure symmetry
        mat[mat > 0] = 1
        G = nx.from_scipy_sparse_array(mat)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            G = pickle.load(f)
    elif path.endswith('.clq') or path.endswith('.dimacs'): 
        # Standard DIMACS format reader
        G = nx.Graph()
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('e'):
                    parts = line.split()
                    u, v = int(parts[1]), int(parts[2])
                    G.add_edge(u-1, v-1) # Convert 1-based to 0-based indices
    else:
        raise ValueError(f"Unsupported file format: {path}")
        
    return G, name

def generate_naive_graph(target_n=600, target_omega=110, target_density=0.3, seed=None):
    """
    Generate a Naive Planted Clique Graph (Random Noise).
    Used as a baseline control group.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        
    G = nx.Graph()
    # 1. Implant Max Clique (Ground Truth)
    clique_nodes = list(range(target_omega))
    G.add_nodes_from(range(target_n))
    for i in range(target_omega):
        for j in range(i+1, target_omega):
            G.add_edge(i, j)
            
    # 2. Add Random Noise Edges to reach target density
    current_m = G.number_of_edges()
    target_m = int(target_density * target_n * (target_n - 1) / 2)
    n_random_edges = target_m - current_m
    
    if n_random_edges > 0:
        # Efficiently add random edges
        cnt = 0
        while cnt < n_random_edges:
            u = random.randint(0, target_n-1)
            v = random.randint(0, target_n-1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                cnt += 1
                
    return G, {
        "name": f"Naive_n{target_n}_w{target_omega}_rho{target_density}",
        "type": "Naive_Planted_Clique",
        "n": target_n,
        "gt_clique": target_omega,
        "density": target_density
    }

def generate_realistic_graph(n_nodes=600, target_density=0.3, clique_size=110, n_decoys=3, seed=None):
    """
    Generate a Realistic Geometric Graph with Decoys.
    Simulates molecular docking energy landscapes with local optima traps.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # 1. Latent Space Embedding (3D)
    positions = np.random.rand(n_nodes, 3)
    dist_matrix = squareform(pdist(positions))
    
    # 2. Geometric Thresholding (Base Graph Structure)
    sorted_dists = np.sort(dist_matrix.flatten())
    sorted_dists = sorted_dists[sorted_dists > 0]
    threshold_idx = int(target_density * len(sorted_dists))
    distance_threshold = sorted_dists[threshold_idx]
    
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    adj_matrix[dist_matrix < distance_threshold] = 1
    np.fill_diagonal(adj_matrix, 0)
    
    # 3. Implant Max Clique (Ground Truth)
    # Simplified: Pick first N nodes. In a real scenario, these would be spatially clustered.
    clique_nodes = list(range(clique_size))
    for i in range(len(clique_nodes)):
        for j in range(i + 1, len(clique_nodes)):
            u, v = clique_nodes[i], clique_nodes[j]
            adj_matrix[u, v] = adj_matrix[v, u] = 1

    # 4. Implant Decoy Cliques (Local Optima)
    current_idx = clique_size
    decoy_size = int(clique_size * 0.7) # Decoys are 70% size of max clique
    
    for _ in range(n_decoys):
        if current_idx + decoy_size < n_nodes:
            decoy_nodes = list(range(current_idx, current_idx + decoy_size))
            for i in range(len(decoy_nodes)):
                for j in range(i + 1, len(decoy_nodes)):
                    if random.random() < 0.95: # 95% density for decoys (almost a clique)
                        u, v = decoy_nodes[i], decoy_nodes[j]
                        adj_matrix[u, v] = adj_matrix[v, u] = 1
            current_idx += decoy_size

    G = nx.from_numpy_array(adj_matrix)
    return G, {
        "name": f"Realistic_n{n_nodes}_w{clique_size}_rho{target_density}_d{n_decoys}",
        "type": "Realistic_Geometric_Decoys", 
        "n": n_nodes, 
        "gt_clique": clique_size,
        "n_decoys": n_decoys,
        "density": target_density
    }