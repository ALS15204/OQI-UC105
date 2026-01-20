#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import networkx as nx
import scipy.io
import time
import json
import random
import math
import pickle

RESULTS_DIR = "/home/kesson/OQI/max_clique_graph_benchmark"
#%% load graph
def load_dimacs_clique(path: str):
    """DIMACS"""
    name = os.path.splitext(os.path.basename(path))[0]

    if path.endswith('.mtx'):
        mat = scipy.io.mmread(path).tocsr()
        # if only upper triangle, converted to symmetry
        mat = mat + mat.T
        mat[mat > 0] = 1
        G = nx.from_scipy_sparse_array(mat)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            G = pickle.load(f)
    else:
        # if DIMACS
        G = nx.Graph()
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                parts = line.split()
                if parts[0] == 'p':
                    n_nodes = int(parts[2])
                    G.add_nodes_from(range(1, n_nodes + 1))
                elif parts[0] == 'e':
                    u, v = int(parts[1]), int(parts[2])
                    G.add_edge(u, v)
        mapping = {node: node - 1 for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

    return G, name

#%% graph properties
def analyze_graph_properties(G, name):
    """graph properties"""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = 2 * m / (n * (n - 1)) if n > 1 else 0

    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0

    # components = list(nx.connected_components(G))
    # largest_component = max(len(c) for c in components) if components else 0

    # clustering_coeff = nx.average_clustering(G) if n > 0 else 0
    
    # theoretical maximum clique upper bound
    import math
    max_possible_clique = int((1 + math.sqrt(1 + 8 * m)) / 2) if m > 0 else 0
    
    properties = {
        'name': name,
        'n_nodes': n,
        'n_edges': m,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        # 'n_components': len(components),
        # 'largest_component': largest_component,
        # 'clustering_coefficient': clustering_coeff,
        'max_possible_clique': max_possible_clique
    }
    
    return properties

#%% FEM
try:
    from FEM.FEM import FEM
except ImportError:
    print("Warning: no FEM, FEM be skipped.")

# energy function
def max_clique_expected_func_v1(J, p):
    """original"""
    if p.dim() == 1:
        p = p.unsqueeze(0)
    
    n = J.shape[0]
    size_term = -torch.sum(p, dim=1)
    p_matrix = p.unsqueeze(2) * p.unsqueeze(1)
    missing_edges = 1 - J - torch.eye(n, device=J.device)
    penalty_strength = 2.0
    penalty_term = penalty_strength * torch.sum(p_matrix * missing_edges, dim=(1, 2))
    total_energy = size_term + penalty_term
    return total_energy

def max_clique_expected_func_v2(J, p):
    """balance penalty"""
    if p.dim() == 1:
        p = p.unsqueeze(0)
    
    # n = J.shape[0]

    size_term = -torch.sum(p, dim=1)

    p_matrix = p.unsqueeze(2) * p.unsqueeze(1)
    actual_connections = torch.sum(p_matrix * J, dim=(1, 2))

    selected_count = torch.sum(p, dim=1)
    ideal_connections = selected_count * (selected_count - 1) / 2
    # penalize the diff between ideal and actual
    missing_connections = torch.relu(ideal_connections - actual_connections)
    
    # penalty_weight
    density = torch.mean(J).item()
    if density > 0.79:
        penalty_weight = 0.5
    elif density > 0.49:
        penalty_weight = 1.0
    else:
        penalty_weight = 2.0
    
    total_energy = size_term + penalty_weight * missing_connections
    return total_energy

def max_clique_expected_func_v3(J, p):
    """degree"""
    if p.dim() == 1:
        p = p.unsqueeze(0)
    
    n = J.shape[0]

    size_term = -torch.sum(p, dim=1)

    degrees = torch.sum(J, dim=1)
    degree_term = -torch.sum(p * degrees.unsqueeze(0), dim=1) / n

    p_matrix = p.unsqueeze(2) * p.unsqueeze(1)
    missing_edges = 1 - J - torch.eye(n, device=J.device)
    
    penalty_strength = 0.5
    penalty_term = penalty_strength * torch.sum(p_matrix * missing_edges, dim=(1, 2)) / (n * n)
    
    total_energy = size_term + 0.1 * degree_term + penalty_term
    return total_energy

def max_clique_infer_func(J, p):
    if p.dim() == 1:
        p = p.unsqueeze(0)
    
    config = (p >= 0.5).float()
    energy = max_clique_expected_func_v1(J, config)  
    
    return config, energy

def extract_clique_from_solutions(G, solutions, energies):
    n = G.number_of_nodes()
    best_clique = []
    best_size = 0
    
    # 1
    for i, sol in enumerate(solutions):
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            selected = torch.where(sol > threshold)[0].tolist()
            if is_valid_clique(G, selected) and len(selected) > best_size:
                best_clique = selected
                best_size = len(selected)
    
    # 2: Greedy
    if best_size == 0:
        for i, sol in enumerate(solutions):

            sorted_indices = torch.argsort(sol, descending=True)
            current_clique = []
            
            for idx in sorted_indices:
                candidate = current_clique + [idx.item()]
                if is_valid_clique(G, candidate):
                    current_clique = candidate

                if len(current_clique) >= min(50, n // 10):
                    break
            
            if len(current_clique) > best_size:
                best_clique = current_clique
                best_size = len(current_clique)
    
    # 3
    if best_size == 0 and len(energies) > 0:
        # Choose lowest energy one, even if it is invalid.
        best_idx = torch.argmin(energies)
        best_sol = solutions[best_idx]

        threshold = torch.mean(best_sol)
        selected = torch.where(best_sol > threshold)[0].tolist()

        sorted_selected = sorted(selected, key=lambda x: best_sol[x].item(), reverse=True)
        current_clique = []
        
        for node in sorted_selected:
            candidate = current_clique + [node]
            if is_valid_clique(G, candidate):
                current_clique = candidate
        
        best_clique = current_clique
        best_size = len(current_clique)
    
    return best_clique, best_size

def solve_with_fem_single(G, params_dic, energy_func, energy_name):
    try:
        coupling_matrix = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        num_nodes = G.number_of_nodes()
        num_interactions = G.number_of_edges()

        fem = FEM.from_couplings(
            'customize',
            num_nodes,
            num_interactions,
            coupling_matrix,
            customize_expected_func=energy_func,
            customize_infer_func=max_clique_infer_func
        )
        
        fem.set_up_solver(
            num_trials=params_dic['N_batch'],
            num_steps=params_dic['N_step'],
            betamin=params_dic['Tmin'],
            betamax=params_dic['Tmax'],
            optimizer=params_dic['opt_mode'],
            learning_rate=params_dic['lr'],
            dev='cpu',
            seed=params_dic['seed']
        )
        
        fem.solver.binary = True
        solutions, energies = fem.solve()

        clique, size = extract_clique_from_solutions(G, solutions, energies)
        is_valid = is_valid_clique(G, clique)
        
        result = {
            'method': 'FEM',
            'clique': clique,
            'size': size,
            'is_valid': is_valid,
            'energy': energies.min().item() if len(energies) > 0 else float('inf'),
            'time': 0,  
            'solutions_found': len(energies),
            'energy_function': energy_name
        }
        
        return result, True
        
    except Exception as e:
        print(f"energy function {energy_name} failed: {e}")
        return None, False

def solve_with_fem(G, params_dic=None):
    """Multi energy func and select the optimal"""
    if params_dic is None:
        n_nodes = G.number_of_nodes()
        density = 2 * G.number_of_edges() / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        if n_nodes > 999:
            # large
            params_dic = {
                'N_batch': 20,     
                'N_step': 5000, #7000   
                'N_rep': 20,        
                "C_grad": 1,
                'seed': 42,
                'lr': 0.1,         
                'Tmin': 0.0001,       
                'Tmax': 0.3,
                'opt_mode': 'rmsprop'
                # 'opt_mode': 'adam'
            }
        elif density > 0.7:
            # density
            params_dic = {
                'N_batch': 30,     
                'N_step': 5000,    
                'N_rep': 30,        
                "C_grad": 1,
                'seed': 42,
                'lr': 0.3,         
                'Tmin': 0.01,       
                'Tmax': 3.0,
                'opt_mode': 'rmsprop'
            }
        else:
            params_dic = {
                'N_batch': 30,     
                'N_step': 3000,    
                'N_rep': 50,        
                "C_grad": 1,
                'seed': 42,
                'lr': 0.5,         
                'Tmin': 0.5,       
                'Tmax': 5.0,
                'opt_mode': 'adam'
            }
    
    start_time = time.time()
    
    energy_functions = [
        (max_clique_expected_func_v1, "origin_func"),
        (max_clique_expected_func_v2, "balance_func"),
        (max_clique_expected_func_v3, "degree_func")
    ]
    
    best_result = None
    all_results = []
    
    print(f"FEM solving:")
    
    for energy_func, energy_name in energy_functions:
        print(f"  try {energy_name}")
        
        result, success = solve_with_fem_single(G, params_dic, energy_func, energy_name)
        
        if success:
            result['time'] = time.time() - start_time
            
            all_results.append(result)
            print(f"    results: value={result['size']}, valid={result['is_valid']}")
            
            # Prioritize efficient solutions with larger clique sizes
            if best_result is None:
                best_result = result
            else:
                if result['is_valid'] and (not best_result['is_valid'] or result['size'] > best_result['size']):
                    best_result = result
                elif not result['is_valid'] and not best_result['is_valid'] and result['size'] > best_result['size']:
                    best_result = result
        else:
            print(f"    failed")
    
    if best_result is None:
        end_time = time.time()
        best_result = {
            'method': 'FEM',
            'clique': [],
            'size': 0,
            'is_valid': False,
            'energy': float('inf'),
            'time': end_time - start_time,
            'solutions_found': 0,
            'energy_function': 'None',
            'error': 'all failed'
        }
    
    if len(all_results) > 1:
        print(f"  FEM compare:")
        for result in all_results:
            print(f"    {result['energy_function']}: value={result['size']}, valid={result['is_valid']}")
        print(f"  final choice: {best_result['energy_function']}")
    
    return best_result

#%% exact method
class ExactMaxClique:
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.best_clique = []
        self.best_size = 0
        self.timeout = 600  # 10 min
        self.start_time = None
        
    def bron_kerbosch(self, R, P, X):
        """Bron-Kerbosch"""
        if time.time() - self.start_time > self.timeout:
            return
            
        if len(P) == 0 and len(X) == 0:
            if len(R) > self.best_size:
                self.best_clique = list(R)
                self.best_size = len(R)
            return
        
        if len(P) > 0:
            pivot = next(iter(P.union(X)))
            neighbors_pivot = set(self.graph.neighbors(pivot))
            
            for v in list(P - neighbors_pivot):
                if time.time() - self.start_time > self.timeout:
                    return
                    
                neighbors_v = set(self.graph.neighbors(v))
                self.bron_kerbosch(R.union([v]), 
                                  P.intersection(neighbors_v), 
                                  X.intersection(neighbors_v))
                P.remove(v)
                X.add(v)
    
    def solve(self):
        """solve max clique"""
        self.start_time = time.time()
        
        R = set()
        P = set(range(self.n))
        X = set()
        
        self.bron_kerbosch(R, P, X)
        
        end_time = time.time()
        time_used = end_time - self.start_time
        timed_out = time_used >= self.timeout
        
        return {
            'method': 'Exact_BronKerbosch',
            'clique': sorted(self.best_clique),
            'size': self.best_size,
            'is_valid': True,  
            'time': time_used,
            'timed_out': timed_out
        }

#%% heuristic method
class HeuristicMaxClique:
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        
    def greedy_heuristic(self):
        """greedy"""
        start_time = time.time()
        
        nodes = sorted(self.graph.nodes(), 
                      key=lambda x: self.graph.degree(x), 
                      reverse=True)
        
        clique = []
        remaining = set(nodes)
        
        while remaining:
            best_node = None
            
            for node in remaining:
                connections = sum(1 for n in clique if self.graph.has_edge(node, n))
                if connections == len(clique):
                    if best_node is None or self.graph.degree(node) > self.graph.degree(best_node):
                        best_node = node
            
            if best_node is None:
                break
                
            clique.append(best_node)
            remaining.remove(best_node)
            remaining = {node for node in remaining if self.graph.has_edge(node, best_node)}
        
        end_time = time.time()
        
        is_valid = is_valid_clique(self.graph, clique)
        
        return {
            'method': 'Heuristic_Greedy',
            'clique': sorted(clique),
            'size': len(clique),
            'is_valid': is_valid,
            'time': end_time - start_time
        }
    
    # def local_search(self, max_iter=3000):
    #     """local search"""
    #     start_time = time.time()
        
    #     # initial from greedy
    #     greedy_result = self.greedy_heuristic()
    #     current_clique = set(greedy_result['clique'])
        
    #     best_clique = current_clique.copy()
    #     best_size = len(best_clique)
        
    #     for iteration in range(max_iter):
    #         # add nodes
    #         candidates = set(range(self.n)) - current_clique
    #         improved = False
            
    #         for node in candidates:
    #             if all(self.graph.has_edge(node, v) for v in current_clique):
    #                 current_clique.add(node)
    #                 improved = True
    #                 break
            
    #         if improved:
    #             if len(current_clique) > best_size:
    #                 best_clique = current_clique.copy()
    #                 best_size = len(best_clique)
    #             continue
            
    #         if len(current_clique) > 1:
    #             node_to_remove = random.choice(list(current_clique))
    #             current_clique.remove(node_to_remove)
        
    #     end_time = time.time()
        
    #     is_valid = is_valid_clique(self.graph, list(best_clique))
        
    #     return {
    #         'method': 'Heuristic_LocalSearch',
    #         'clique': sorted(best_clique),
    #         'size': len(best_clique),
    #         'is_valid': is_valid,
    #         'time': end_time - start_time
    #     }
    def local_search(self, max_iter=5000, initial_temperature=100.0, cooling_rate=0.99):
        """local search with simulated annealing"""
        start_time = time.time()

        greedy_result = self.greedy_heuristic()
        current_clique = set(greedy_result['clique'])
        current_size = len(current_clique)
        
        best_clique = current_clique.copy()
        best_size = current_size
        
        temperature = initial_temperature
        
        for iteration in range(max_iter):

            candidates = set(range(self.n)) - current_clique
            improved = False
            
            candidates_list = list(candidates)
            random.shuffle(candidates_list)
            
            for node in candidates_list:
                if all(self.graph.has_edge(node, v) for v in current_clique):
                    current_clique.add(node)
                    current_size += 1
                    improved = True
                    break

            if improved:
                if current_size > best_size:
                    best_clique = current_clique.copy()
                    best_size = current_size
                continue
            

            if current_size >= 2:
                external_candidates = set(range(self.n)) - current_clique
                improved_by_swap = False
                
                for node_in in random.sample(list(external_candidates), min(10, len(external_candidates))):

                    non_neighbors_in_clique = [v for v in current_clique if not self.graph.has_edge(node_in, v)]
                    
                    if len(non_neighbors_in_clique) == 1:

                        node_out = non_neighbors_in_clique[0]
                        new_clique = current_clique - {node_out} | {node_in}
                        

                        if is_valid_clique(self.graph, list(new_clique)):

                            delta_energy = -(len(new_clique) - current_size)
                            
                         
                            if delta_energy <= 0 or random.random() < math.exp(-delta_energy / temperature):
                                current_clique = new_clique
                                current_size = len(current_clique)
                                improved_by_swap = True
                                
                                if current_size > best_size:
                                    best_clique = current_clique.copy()
                                    best_size = current_size
                                break
                    
                    elif len(non_neighbors_in_clique) == 0:
                       
                        current_clique.add(node_in)
                        current_size += 1
                        improved_by_swap = True
                        
                        if current_size > best_size:
                            best_clique = current_clique.copy()
                            best_size = current_size
                        break
                
                if improved_by_swap:
                    continue
            

            if current_size > 1 and random.random() < 0.1: 
                node_to_remove = random.choice(list(current_clique))
                current_clique.remove(node_to_remove)
                current_size -= 1

            temperature *= cooling_rate
            
            if iteration % 500 == 0 and iteration > 0:
                
                if random.random() < 0.3:  
                    if best_size > 0:
                        
                        current_clique = best_clique.copy()
                        current_size = best_size
                    else:

                        current_clique = self.generate_random_clique()
                        current_size = len(current_clique)
        
        end_time = time.time()
        
        is_valid = is_valid_clique(self.graph, list(best_clique))
        
        return {
            'method': 'Heuristic_LocalSearch_SA',
            'clique': sorted(best_clique),
            'size': len(best_clique),
            'is_valid': is_valid,
            'time': end_time - start_time
        }
    
    def generate_random_clique(self):
        nodes = list(range(self.n))
        random.shuffle(nodes)
        
        clique = set()
        for node in nodes:
            if all(self.graph.has_edge(node, v) for v in clique):
                clique.add(node)
        
        return clique

#%% others
def is_valid_clique(G, nodes):
    """is clique"""
    if len(nodes) <= 1:
        return True
    return all(G.has_edge(nodes[i], nodes[j]) 
               for i in range(len(nodes)) 
               for j in range(i+1, len(nodes)))

def save_results(results, graph_properties, filename=None):
    """save to JSON"""
    # if filename is None:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"max_clique_results_{timestamp}.json"
    if filename is None:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_name = graph_properties['name']
        filename = f"max_clique_{graph_name}.json"
    
    # 
    filepath = os.path.join(RESULTS_DIR, filename)        
    data = {
        'graph_properties': graph_properties,
        'results': results,
        #'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"saved to: {filepath}")
    return filepath

def load_results(filename):
    """load"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

#%% Main function - Unified solution framework
def solve_max_clique_unified(graph_path, methods=None, save_output=True):
    """
    Maximum Clique Solution Framework
    
    param:
    - graph_path
    - methods: Use all methods by default
    - save_output: True
    """
    
    if methods is None:
        methods = ['FEM', 'Exact', 'Greedy', 'LocalSearch']
    
    # load
    G, name = load_dimacs_clique(graph_path)
    print(f"\n{'='*60}")
    print(f"solve: {name}")
    print(f"{'='*60}")
    
    # analysis
    graph_properties = analyze_graph_properties(G, name)
    print("\n graph properties:")
    for key, value in graph_properties.items():
        print(f"  {key}: {value}")
    
    results = []
    
    # FEM
    if 'FEM' in methods:
        print(f"\n[1/{len(methods)}] Solving using the FEM...")
        fem_result = solve_with_fem(G)
        results.append(fem_result)
        print(f"  FEM results: value={fem_result['size']}, valid={fem_result['is_valid']}, time={fem_result['time']:.2f}s")
    
    # Exact
    if 'Exact' in methods:
        print(f"\n[2/{len(methods)}] Solving using the Exact...")
        exact_solver = ExactMaxClique(G)
        exact_result = exact_solver.solve()
        results.append(exact_result)
        print(f"  Exact results: value={exact_result['size']}, time={exact_result['time']:.2f}s")
        if exact_result['timed_out']:
            print("Note: Exact method timed out, and the result is the best solution found so far.")
    
    # Greedy
    if 'Greedy' in methods:
        print(f"\n[3/{len(methods)}] Solving using the Greedy...")
        heuristic_solver = HeuristicMaxClique(G)
        greedy_result = heuristic_solver.greedy_heuristic()
        results.append(greedy_result)
        print(f"  Greedy results: value={greedy_result['size']}, valid={greedy_result['is_valid']}, time={greedy_result['time']:.2f}s")
    
    # Local search
    if 'LocalSearch' in methods:
        print(f"\n[4/{len(methods)}] Solving using the Local search...")
        local_result = heuristic_solver.local_search()
        results.append(local_result)
        print(f"  Local search results: value={local_result['size']}, valid={local_result['is_valid']}, time={local_result['time']:.2f}s")
    
    # results
    print(f"\n{'='*60}")
    print("summary:")
    print(f"{'='*60}")
    
    valid_results = [r for r in results if r['is_valid']]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['size'])
        print(f"best result: {best_result['method']} - value={best_result['size']}")
    else:
        print("Warning: No valid clique found.")
        best_result = max(results, key=lambda x: x['size'])
    
    print("\n Table:")
    for result in results:
        status = "valid" if result['is_valid'] else "invalid"
        print(f"  {result['method']:20} | solution: {result['size']:3d} | {status:4} | time: {result['time']:7.2f}s")
    
    if save_output:
        filepath = save_results(results, graph_properties)
    
    return results, graph_properties

def compare_multiple_graphs(graph_paths, methods=None):
    """compare"""
    all_results = {}
    
    for graph_path in graph_paths:
        G, name = load_dimacs_clique(graph_path)
        graph_properties = analyze_graph_properties(G, name)
        
        print(f"\n graphs: {name} (n={graph_properties['n_nodes']}, m={graph_properties['n_edges']})")
        
        results = []
        
        # add methods
        if methods is None or 'FEM' in methods:
            fem_result = solve_with_fem(G)
            results.append(fem_result)
        
        if methods is None or 'Exact' in methods:
            exact_solver = ExactMaxClique(G)
            exact_result = exact_solver.solve()
            results.append(exact_result)
        
        if methods is None or 'Greedy' in methods:
            heuristic_solver = HeuristicMaxClique(G)
            greedy_result = heuristic_solver.greedy_heuristic()
            results.append(greedy_result)
        
        all_results[name] = {
            'properties': graph_properties,
            'results': results
        }
    
    return all_results


#%% main
if __name__ == "__main__":
    # graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/c-fat200-1.mtx"
    # graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/brock200-1.mtx"
    # graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/gen200-p0-9-44.mtx"
    # graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/MANN-a45.mtx"
    graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/mole_n600_omega110_den0.3.pkl"
    # graph_path = "/home/kesson/OQI/max_clique_graph_benchmark/random_n2000_p0.3.pkl"
    results, properties = solve_max_clique_unified(graph_path)
    # select methods
    # results, properties = solve_max_clique_unified(
    #     graph_path, 
    #     methods=['FEM'],  
    #     save_output=False
    # )
    
    # compare
    # graph_paths = [
    #     "graph1.mtx",
    #     "graph2.mtx",
    #     "graph3.mtx"
    # ]
    # all_results = compare_multiple_graphs(graph_paths)
    # plot_comparison(all_results)
#%% load
def print_simple_table(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    print("Table:")
    for result in results:
        method = result.get('method', 'Unknown')
        size = result.get('size', 0)
        is_valid = result.get('is_valid', False)
        time_val = result.get('time', 0)
        
        valid_str = "valid" if is_valid else "invalid"
        time_str = f"{time_val:.2f}s"
        
        print(f"  {method:<20} | solution: {size:>3d} | {valid_str} | time: {time_str:>8}")
