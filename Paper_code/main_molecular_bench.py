import pandas as pd
import os
from src.graph_gen import generate_realistic_graph
from src.solvers import MaxCliqueSolver
from src.visualization import plot_benchmark_results

def run_molecular_benchmark():
    results = []
    os.makedirs("results", exist_ok=True)
    
    # Extended Test Scenarios for Molecular-like Graphs
    configs = [
        # Group 1: Sparse graphs (Easy)
        {"N": 600, "Omega": 50,  "Density": 0.2, "Decoys": 3, "Desc": "Small Sparse"},
        
        # Group 2: Dense graphs (Harder, more geometric noise)
        {"N": 600, "Omega": 80,  "Density": 0.4, "Decoys": 5, "Desc": "Small Dense"},
        
        # Group 3: Large Standard
        {"N": 1000, "Omega": 100, "Density": 0.3, "Decoys": 5,  "Desc": "Large Standard"},
        
        # Group 4: Stress Test (High Decoys, Dense)
        {"N": 1000, "Omega": 150, "Density": 0.5, "Decoys": 10, "Desc": "Large Dense + Many Decoys"},
        
        # Group 5: Complex Pocket Simulation
        {"N": 2000, "Omega": 200, "Density": 0.3, "Decoys": 8, "Desc": "Complex Pocket"},
    ]
    
    print(f"=== Starting Molecular Graph Benchmark ({len(configs)} configs) ===")
    
    for i, cfg in enumerate(configs):
        print(f"\n[Test {i+1}] {cfg['Desc']}: N={cfg['N']}, Omega={cfg['Omega']}, Density={cfg['Density']}")
        
        # Generate
        G, info = generate_realistic_graph(
            n_nodes=cfg['N'], 
            target_density=cfg['Density'],
            clique_size=cfg['Omega'], 
            n_decoys=cfg['Decoys'],
            seed=42 + i
        )
        
        solver = MaxCliqueSolver(G)
        
        # 1. Greedy
        res_greedy = solver.solve_greedy()
        results.append({'Instance': info['name'], 'Method': 'Greedy', 'Size': res_greedy['size']})
        
        # 2. Local Search (Increased iter for hard cases)
        res_sa = solver.solve_localsearch_sa(max_iter=8000) 
        results.append({'Instance': info['name'], 'Method': 'LocalSearch', 'Size': res_sa['size']})
        
        # 3. Ground Truth
        results.append({'Instance': info['name'], 'Method': 'Target (GT)', 'Size': cfg['Omega']})
        
        print(f"   -> Greedy: {res_greedy['size']}, SA: {res_sa['size']} (Target: {cfg['Omega']})")

    # Save & Plot
    df = pd.DataFrame(results)
    csv_path = "results/molecular_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    plot_benchmark_results(df, "results/molecular_benchmark_plot.png", title="Molecular Graph Benchmark")

if __name__ == "__main__":
    run_molecular_benchmark()