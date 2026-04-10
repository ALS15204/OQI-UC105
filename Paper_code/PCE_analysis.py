import numpy as np
import networkx as nx
import random
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from itertools import combinations, product

# ==========================================
# 1. Realistic Graph Generator
# ==========================================
def generate_realistic_graph(n_nodes=40, target_density=0.35, clique_size=8, n_decoys=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    positions = np.random.rand(n_nodes, 3)
    dist_matrix = squareform(pdist(positions))
    
    sorted_dists = np.sort(dist_matrix.flatten())
    sorted_dists = sorted_dists[sorted_dists > 0]
    distance_threshold = sorted_dists[int(target_density * len(sorted_dists))]
    
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    adj_matrix[dist_matrix < distance_threshold] = 1
    np.fill_diagonal(adj_matrix, 0)
    
    gt_nodes = list(range(clique_size))
    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            adj_matrix[gt_nodes[i], gt_nodes[j]] = 1
            adj_matrix[gt_nodes[j], gt_nodes[i]] = 1

    current_idx = clique_size
    decoy_groups = []
    
    for d_idx in range(n_decoys):
        d_size = int(clique_size * (0.8 - d_idx * 0.1))
        if current_idx + d_size < n_nodes and d_size > 2:
            d_nodes = list(range(current_idx, current_idx + d_size))
            for i in range(len(d_nodes)):
                for j in range(i + 1, len(d_nodes)):
                    if random.random() < 0.95:
                        adj_matrix[d_nodes[i], d_nodes[j]] = 1
                        adj_matrix[d_nodes[j], d_nodes[i]] = 1
            decoy_groups.append(d_nodes)
            current_idx += d_size
            
    return nx.from_numpy_array(adj_matrix), gt_nodes, decoy_groups

# ==========================================
# 2. Greedy Compiler & Sparse Mapping
# ==========================================
def commutes(p1, p2):
    return sum(1 for c1, c2 in zip(p1, p2) if c1 != 'I' and c2 != 'I' and c1 != c2) % 2 == 0

def generate_greedy_pauli_pool(N, V, max_weight):
    candidates = []
    for w in range(1, min(N + 1, max_weight + 1)):
        for comb in combinations(range(N), w):
            for p_chars in product(['X', 'Y', 'Z'], repeat=w):
                chars = ['I'] * N
                for idx, char in zip(comb, p_chars): chars[idx] = char
                candidates.append("".join(chars))
                
    random.shuffle(candidates)
    pool = [candidates.pop(0)]
    
    while len(pool) < V:
        best_score = -1
        best_idx = -1
        for i, cand in enumerate(candidates):
            score = sum(1 for p in pool if commutes(cand, p))
            if score > best_score:
                best_score = score
                best_idx = i
                if best_score == len(pool): break 
        pool.append(candidates.pop(best_idx))
    return pool

def get_pauli_sparse(p_str, N):
    dim = 1 << N
    row = np.arange(dim, dtype=np.int32)
    col = np.copy(row)
    data = np.ones(dim, dtype=np.complex128)
    for i, char in enumerate(p_str):
        bit = N - 1 - i
        state_bit = (row >> bit) & 1
        if char == 'X': col ^= (1 << bit)
        elif char == 'Y': col ^= (1 << bit); data *= np.where(state_bit == 0, 1j, -1j)
        elif char == 'Z': data *= np.where(state_bit == 0, 1.0, -1.0)
    return sp.csr_matrix((data, (row, col)), shape=(dim, dim))

# ==========================================
# 3. Exact Krylov Subspace Solver
# ==========================================
class FinalPCEAnalyzer:
    def solve_ensemble(self, V, N, density, trials=20, base_seed=42):
        fail_count = 0
        margins, stds = [], []
        
        for t in range(trials):
            current_seed = base_seed + t
            G, gt_nodes, decoy_groups = generate_realistic_graph(n_nodes=V, target_density=density, clique_size=8, n_decoys=2, seed=current_seed)
            
            paulis = generate_greedy_pauli_pool(N, V, max_weight=2)
            
            dim = 1 << N
            I_mat = sp.eye(dim, dtype=np.complex128, format='csr')
            x_ops = [(I_mat - get_pauli_sparse(p, N)) * 0.5 for p in paulis]
            
            X_sum = sum(x_ops)
            H = -1.0 * X_sum
            
            non_edges = list(nx.non_edges(G))
            H_penalty = sp.csr_matrix((dim, dim), dtype=np.complex128)
            for u, v in non_edges:
                term = x_ops[u] @ x_ops[v]
                H_penalty = H_penalty + 0.5 * (term + term.conj().T)
            
            H = H + 2.0 * H_penalty
            
            evals, evecs = eigsh(H, k=1, which='SA', tol=1e-5)
            gs_state = evecs[:, 0]
            
            node_probs = [np.real(np.vdot(gs_state, x_ops[v] @ gs_state)) for v in range(V)]
            
            gt_mean = np.mean([node_probs[i] for i in gt_nodes])
            gt_std = np.std([node_probs[i] for i in gt_nodes])
            decoy_means = [np.mean([node_probs[i] for i in dg]) for dg in decoy_groups]
            top_decoy = max(decoy_means) if decoy_means else 0
            
            margin = gt_mean - top_decoy
            margins.append(margin)
            stds.append(gt_std)
            
            if margin < 0:
                fail_count += 1
                
        return (fail_count / trials) * 100, np.mean(margins), np.mean(stds)

def print_paper_table_row(label_str, fail_rate, margin, std, capacity_note=""):
    if fail_rate >= 50.0:
        status = "Global (Decoy Trap)"
    elif std > 0.05:
        status = "Local (MD-Fixable)"
    else:
        status = "Perfect Topology"
        
    margin_str = f"+{margin:.4f}" if margin >= 0 else f"{margin:.4f}"
    print(f" {label_str:<15} | {fail_rate:>10.1f}% | {margin_str:<10} | {std:<10.4f} | {status:<20} {capacity_note}")

# ==========================================
# 4. Main Execution (Dual Tables)
# ==========================================
if __name__ == "__main__":
    V = 40
    TRIALS = 20
    analyzer = FinalPCEAnalyzer()
    
    print("\n" + "="*105)
    print(f" TABLE I: The Deterministic Qubit Compression Bound (Fixed Density ρ=0.35, Graph V={V}, w<=2)")
    print(" (Testing the Pigeonhole Principle: Max Commuting Operators Capacity C = N + N(N-1)/2)")
    print("="*105)
    print(f" {'Qubits (N)':<15} | {'Failure %':<10} | {'GT Margin':<10} | {'GT Fluct(σ)':<10} | {'Primary Error Mode':<20} | {'Theoretical Capacity'}")
    print("-" * 105)
    
    # 測試 N 從 7 掃到 10，觀察容量突破 40 前後的變化
    N_list = [7, 8, 9, 10]
    for n in N_list:
        t0 = time.time()
        capacity = n + (n * (n - 1)) // 2
        cap_note = f"| C={capacity} (< {V}, Frustrated)" if capacity < V else f"| C={capacity} (> {V}, Safe)"
        
        fail_rate, avg_margin, avg_std = analyzer.solve_ensemble(V=V, N=n, density=0.35, trials=TRIALS)
        time_str = f"N={n} ({time.time()-t0:.1f}s)"
        print_paper_table_row(time_str, fail_rate, avg_margin, avg_std, cap_note)

    print("\n" + "="*105)
    print(f" TABLE II: The Flexibility Wall vs. Graph Density (Fixed Qubits N=10, Graph V={V}, w<=2)")
    print(f" (Hardware theoretically has sufficient capacity: C=55 > V=40. Testing topology collapse.)")
    print("="*105)
    print(f" {'Density (ρ)':<15} | {'Failure %':<10} | {'GT Margin':<10} | {'GT Fluct(σ)':<10} | {'Primary Error Mode':<20}")
    print("-" * 105)
    
    # 測試密度從 0.15 掃到 0.45
    densities = [0.15, 0.25, 0.35, 0.45]
    for d in densities:
        t0 = time.time()
        fail_rate, avg_margin, avg_std = analyzer.solve_ensemble(V=V, N=10, density=d, trials=TRIALS)
        time_str = f"ρ={d:.2f} ({time.time()-t0:.1f}s)"
        print_paper_table_row(time_str, fail_rate, avg_margin, avg_std)
        
    print("\n" + "="*105)