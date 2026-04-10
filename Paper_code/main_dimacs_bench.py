import os
import pandas as pd
from src.graph_gen import load_graph
from src.solvers import MaxCliqueSolver

def run_dimacs_benchmark():
    data_dir = "data/dimacs"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please move .mtx files there.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".mtx")]
    results = []

    print(f"=== Benchmarking Standard DIMACS ({len(files)} files) ===")

    for f in files:
        path = os.path.join(data_dir, f)
        try:
            G, name = load_graph(path)
            print(f"\nProcessing {name} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
            
            solver = MaxCliqueSolver(G)
            
            # 1. Greedy
            res_greedy = solver.solve_greedy()
            results.append({'Instance': name, 'Method': 'Greedy', 'Size': res_greedy['size'], 'Time': res_greedy['time']})
            
            # 2. Local Search
            res_sa = solver.solve_localsearch_sa(max_iter=5000)
            results.append({'Instance': name, 'Method': 'LocalSearch', 'Size': res_sa['size'], 'Time': res_sa['time']})
            
            print(f"   -> Greedy: {res_greedy['size']}, SA: {res_sa['size']}")
            
        except Exception as e:
            print(f"   -> Error loading {f}: {e}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/dimacs_benchmark_results.csv", index=False)
        print("\nDIMACS Results saved.")

if __name__ == "__main__":
    run_dimacs_benchmark()