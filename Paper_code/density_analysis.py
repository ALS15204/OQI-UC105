import os
import time
import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from rdkit import Chem
from rdkit.Chem import AllChem

# Assuming src.mol_prep is available in your environment
from src.mol_prep import (
    extract_submol, 
    get_pocket_center_generic, 
    extract_pocket_by_center, 
    build_quantum_pharmacophore_graph
)

# =========================================================
# Helper: Download & Load PDB
# =========================================================
def get_mol_from_pdb_id(pdb_id, save_dir="data_graph_bench"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{pdb_id}.pdb")
    
    if not os.path.exists(path):
        print(f"Downloading {pdb_id}...")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f: 
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
            return None
    
    if os.path.exists(path):
        mol_raw = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        return mol_raw
    return None

# =========================================================
# Core: 2D Quantum Graph Profiling Function
# =========================================================
def profile_quantum_graph_2d(case_name, ligand_mol, full_receptor_mol, center_coords, cutoffs, tolerances):
    print(f"\n{'='*50}")
    print(f" Profiling 2D Phase Space: {case_name}")
    print(f"{'='*50}")
    
    results = []
    
    for cutoff in cutoffs:
        print(f"\n[Extracing Pocket] Cut-off Radius = {cutoff} Å")
        try:
            rec_pocket = extract_pocket_by_center(full_receptor_mol, center_coords, cutoff=cutoff)
        except Exception as e:
            print(f"  -> [Error] Failed to extract pocket at cutoff {cutoff}: {e}")
            continue

        for tol in tolerances:
            print(f"  -> Testing Max Tolerance = {tol} Å", end="", flush=True)
            t0 = time.time()
            try:
                G, node_map, lig_feats, rec_feats, Q_dict = build_quantum_pharmacophore_graph(
                    ligand_mol, rec_pocket, 
                    max_tolerance=tol, 
                    alpha=1.0,      
                    reward=-2.0,    
                    penalty=100.0   
                )
            except Exception as e:
                print(f" | [Error] Graph build failed: {e}")
                continue

            build_time = time.time() - t0
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            if num_nodes == 0:
                print(" | [Warning] Empty graph.")
                continue
                
            density = nx.density(G)
            
            results.append({
                "Case": case_name,
                "Cut-off (A)": cutoff,
                "Tolerance (A)": tol,
                "Nodes (Qubits)": num_nodes,
                "Edges": num_edges,
                "Graph Density": density,
                "Build Time (s)": build_time
            })
            print(f" | Nodes: {num_nodes}, Density: {density:.3f}, Time: {build_time:.2f}s")

    return results

# =========================================================
# Visualization: Flexibility Wall Phase Diagram
# =========================================================
def plot_phase_diagram(df, case_name, save_path=None):
    # Pivot data for heatmap (Rows: Cut-off, Cols: Tolerance)
    df_density = df.pivot(index="Cut-off (A)", columns="Tolerance (A)", values="Graph Density")
    df_nodes = df.pivot(index="Cut-off (A)", columns="Tolerance (A)", values="Nodes (Qubits)")
    
    # Sort Y-axis descending so larger cutoffs are at the top
    df_density = df_density.sort_index(ascending=False)
    df_nodes = df_nodes.sort_index(ascending=False)

    # Create custom annotations (Density + Node count in each cell)
    annot_matrix = np.empty(df_density.shape, dtype=object)
    for i in range(df_density.shape[0]):
        for j in range(df_density.shape[1]):
            val_dens = df_density.iloc[i, j]
            val_nodes = df_nodes.iloc[i, j]
            if pd.isna(val_dens) or pd.isna(val_nodes):
                annot_matrix[i, j] = "N/A"
            else:
                annot_matrix[i, j] = f"{val_dens:.2f}\n(n={int(val_nodes)})"

    # IEEE Standard Font Settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 8
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Draw heatmap
    sns.heatmap(
        df_density, 
        annot=annot_matrix, 
        fmt="", 
        cmap="YlOrRd", 
        linewidths=.5, 
        ax=ax,
        cbar_kws={'label': r'Graph Density, $\rho$'}
    )

    ax.set_xlabel(r"Rigidity Tolerance, $\tau$ (Å)")
    ax.set_ylabel(r"Pocket Cut-off (Å)")
    ax.set_title(f"Biochemical Compression Phase Diagram ({case_name})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# =========================================================
# Main Execution Block
# =========================================================
if __name__ == "__main__":
    tolerances_to_test = [2.0, 3.0, 4.0, 5.0, 6.0]
    cutoffs_to_test = [6.0, 8.0, 10.0, 12.0, 14.0]
    
    all_results = []
    base_dir = "data_graph_bench"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Setup Data: Self-Docking (2OI0)
    print("--- Loading 2OI0 (Self-Docking) ---")
    mol_2oi0 = get_mol_from_pdb_id("2OI0", save_dir=base_dir)
    if mol_2oi0:
        lig_2oi0, _ = extract_submol(mol_2oi0)
        lig_2oi0 = AllChem.AddHs(lig_2oi0, addCoords=True)
        center_2oi0, _ = get_pocket_center_generic(mol_2oi0, None)
        
        res_2oi0 = profile_quantum_graph_2d(
            "2OI0 Self-Docking", 
            lig_2oi0, 
            mol_2oi0, 
            center_2oi0, 
            cutoffs_to_test, 
            tolerances_to_test
        )
        all_results.extend(res_2oi0)


    # =========================================================
    # Report & Visualization
    # =========================================================
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print("\n" + "="*80)
        print(" 2D QUANTUM GRAPH GENERATION BENCHMARK SUMMARY")
        print("="*80)
        print(df_results.to_string(index=False))
        print("="*80)
        
        # Plot Phase Diagram ONLY for 1UY6
        df_1uy6 = df_results[df_results["Case"] == "2OI0 Self-Docking"]
        if not df_1uy6.empty:
            plot_phase_diagram(df_1uy6, "2OI0 Self-Docking", save_path="phase_diagram.png")
            
    else:
        print("No results generated. Check data loading and logic.")