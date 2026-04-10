import os
import time
import subprocess
import requests
import shutil
import pandas as pd
import numpy as np
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem

# --- 1. Custom Modules ---
from src.solvers import MaxCliqueSolver
from src.mol_prep import (
    extract_submol, 
    get_pocket_center_generic, 
    extract_pocket_by_center, 
    build_pharmacophore_graph, 
    enforce_physical_distance,
    build_quantum_pharmacophore_graph
)

try:
    from src.openmm_md import run_short_md_evaluation
    MD_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] MD evaluation not available: {e}")
    MD_AVAILABLE = False

# --- 2. Local Helpers ---
def download_pdb(pdb_id, save_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"   [Download] Fetching {pdb_id}...")
    try:
        r = requests.get(url, timeout=15)
        with open(save_path, "wb") as f: f.write(r.content)
    except Exception as e: 
        print(f"   [Error] Download failed: {e}")

def convert_to_pdbqt(pdb_path, pdbqt_path, flexible=False):
    try: 
        cmd = ["obabel", pdb_path, "-O", pdbqt_path, "-d"] 
        if flexible: cmd.append("-xr")
        subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)
    except: pass

def get_mol_center(mol):
    conf = mol.GetConformer()
    pos = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    return np.mean(pos, axis=0)

def generate_docked_pdb(clique, node_map, lig_feats, rec_feats, ligand_mol, output_path):
    if len(clique) < 3: return None
    src = np.array([lig_feats[node_map[n][0]]['c'] for n in clique])
    dst = np.array([rec_feats[node_map[n][1]]['c'] for n in clique])
    
    c1, c2 = np.mean(src, 0), np.mean(dst, 0)
    src -= c1; dst -= c2
    H = src.T @ dst
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[2,:] *= -1; R = Vt.T @ U.T
    t = c2 - R @ c1
    
    new_mol = Chem.Mol(ligand_mol)
    conf = new_mol.GetConformer()
    for i in range(new_mol.GetNumAtoms()):
        p = np.array(conf.GetAtomPosition(i))
        new_p = R @ p + t
        conf.SetAtomPosition(i, list(new_p))
    
    Chem.MolToPDBFile(new_mol, output_path)
    return new_mol

def prepare_ligand_for_smina(mol, output_path):
    w = Chem.SDWriter(output_path)
    w.write(mol)
    w.close()
    return True

def run_smina_sdf(receptor_path, ligand_path, output_path, center=None, size=(20, 20, 20), extra_args=""):
    cmd = ["smina", "-r", receptor_path, "-l", ligand_path, "-o", output_path, "--cpu", "4", "--quiet"]
    if center is not None:
        cmd.extend(["--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
                    "--size_x", f"{size[0]}", "--size_y", f"{size[1]}", "--size_z", f"{size[2]}"])
    else:
        cmd.extend(["--autobox_ligand", ligand_path])
        
    if extra_args: cmd.extend(extra_args.split())

    try:
        subprocess.run(cmd, check=True)
        score = 0.0
        if os.path.exists(output_path):
            suppl = Chem.SDMolSupplier(output_path, sanitize=False)
            for mol in suppl:
                if mol and mol.HasProp("minimizedAffinity"):
                    s = float(mol.GetProp("minimizedAffinity"))
                    if abs(s) > 0.001: score = s; break
        return score, os.path.exists(output_path)
    except Exception as e:
        print(f"   [Smina Error] {e}")
        return 0.0, False

# =====================================================================
# --- Stage 3: Standalone MM-GBSA Evaluator (Runs AFTER MD) ---
# =====================================================================
def calculate_mmgbsa_from_trajectory(complex_pdb_path, dcd_path, ligand_mol, base_dir, prefix):
    """
    Strips explicit water from the trajectory, applies Implicit Solvent (igb),
    and calculates the structural potential energy across frames.
    """
    try:
        from openmm.app import PDBFile, ForceField, Modeller, NoCutoff
        from openmm import LangevinMiddleIntegrator, Context
        import openmm.unit as unit
        from openmmforcefields.generators import GAFFTemplateGenerator
        from openff.toolkit.topology import Molecule
    except ImportError:
        print(" [Error] Required OpenMM packages for MM-GBSA are missing.")
        return None

    print(f" [MM-GBSA] Loading explicit trajectory: {dcd_path}...")
    
    # 1. Strip Water and Ions using MDTraj
    traj = md.load_dcd(dcd_path, top=complex_pdb_path)
    selection_query = "(not water) and (not resname HOH) and (not resname NA) and (not resname CL) and (not element Na) and (not element Cl)"
    dry_indices = traj.topology.select(selection_query)
    
    if len(dry_indices) == 0:
        print(" [MM-GBSA Error] Stripping water failed (empty selection).")
        return None
        
    dry_traj = traj.atom_slice(dry_indices)
    
    # Save dry files temporarily for OpenMM to read
    dry_pdb_path = os.path.join(base_dir, f"{prefix}_dry.pdb")
    dry_traj[0].save_pdb(dry_pdb_path)
    
    # 2. Setup OpenMM Implicit System
    print(" [MM-GBSA] Setting up Implicit Solvent System (Amber14 + igb)...")
    pdb = PDBFile(dry_pdb_path)
    
    # Prepare OpenFF Molecule for GAFF (required for ligand parameters)
    fixed_rdkit_mol = Chem.AddHs(ligand_mol, addCoords=True)
    off_mol = Molecule.from_rdkit(fixed_rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    
    forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
    gaff = GAFFTemplateGenerator(molecules=off_mol)
    forcefield.registerTemplateGenerator(gaff.generator)
    
    try:
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)
        integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        context = Context(system, integrator)
    except Exception as e:
        print(f" [MM-GBSA Error] System creation failed: {e}")
        return None

    energies = []
    print(" [MM-GBSA] Analyzing frames...")
    
    # 3. Evaluate Energy for each frame
    for i in range(dry_traj.n_frames):
        positions = dry_traj.xyz[i] * 10.0 # Convert from nm (mdtraj) to Angstroms/OpenMM units
        context.setPositions(positions * unit.angstroms)
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        energies.append(pot_energy)
        
        if i % 50 == 0 or i == dry_traj.n_frames - 1:
            print(f"   Frame {i}: {pot_energy:.2f} kcal/mol")
        
    avg_energy = np.mean(energies)
    std_energy = np.std(energies)
    print(f" [MM-GBSA] Average Complex Energy (Implicit): {avg_energy:.2f} ± {std_energy:.2f} kcal/mol")
    
    return avg_energy

# =====================================================================
# --- Stage 1 & 2: Docking and MD Pipeline ---
# =====================================================================
def run_laccase_phenol_benchmark():
    receptor_id = "1GYC" 
    ligand_name = "Phenol"
    phenol_smiles = "c1ccccc1O"
    
    print(f"=== Stage 1: Preparation & Docking | {ligand_name} -> Laccase ({receptor_id}) ===")
    base_dir = "data_laccase"
    os.makedirs(base_dir, exist_ok=True)
    
    path_rec = os.path.join(base_dir, f"{receptor_id}.pdb")
    if not os.path.exists(path_rec): download_pdb(receptor_id, path_rec)
        
    mol_rec_raw = Chem.MolFromPDBFile(path_rec, removeHs=False, sanitize=False)
    pocket_center, source_type = get_pocket_center_generic(mol_rec_raw, None)
    is_metal_pocket = "Metal" in source_type
    
    rec_clean_pdb = os.path.join(base_dir, "clean_receptor.pdb")
    shutil.copy(path_rec, rec_clean_pdb)
    rec_qt = os.path.join(base_dir, "receptor.pdbqt")
    convert_to_pdbqt(path_rec, rec_qt)
    receptor_mol_pocket = extract_pocket_by_center(mol_rec_raw, pocket_center, cutoff=25.0)
    
    # Prepare Ligand
    phenol_mol = Chem.MolFromSmiles(phenol_smiles)
    phenol_mol = Chem.AddHs(phenol_mol)
    AllChem.EmbedMolecule(phenol_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(phenol_mol)
    
    results = []
    methods = ["Graph_Greedy", "Graph_SA"]
    tolerances = [3.5] if is_metal_pocket else [2.0, 2.5] 

    for method_name in methods:
        print(f"\n>>> Method: {method_name}")
        t0_total = time.time()
        best_clique = []
        best_data = None
        
        for tol in tolerances:
            G, nmap, lfeats, rfeats, Q_dict = build_quantum_pharmacophore_graph(phenol_mol, receptor_mol_pocket, max_tolerance=tol)
            if len(G.nodes()) == 0: continue
            
            solver = MaxCliqueSolver(G)
            res = solver.solve_sa(iterations=2000) if method_name == "Graph_SA" else solver.solve_greedy()
            
            if len(res['clique']) > len(best_clique):
                best_clique = res['clique']
                best_data = (nmap, lfeats, rfeats)
                if len(best_clique) >= 5: break 
        
        if best_clique:
            nmap, lfeats, rfeats = best_data
            docked_mol = generate_docked_pdb(best_clique, nmap, lfeats, rfeats, phenol_mol, 
                                             os.path.join(base_dir, f"pose_{method_name}.pdb"))
            
            # --- Smina Refinement ---
            docked_mol_safe = enforce_physical_distance(docked_mol, pocket_center, is_metal=is_metal_pocket)
            in_sdf = os.path.join(base_dir, f"{method_name}_in.sdf")
            out_sdf = os.path.join(base_dir, f"{method_name}_out.sdf")
            prepare_ligand_for_smina(docked_mol_safe, in_sdf)
            
            energy, success = run_smina_sdf(
                rec_qt, in_sdf, out_sdf, 
                center=get_mol_center(docked_mol_safe), size=(10, 10, 10), 
                extra_args="--exhaustiveness 8"
            )
            
            final_mol = Chem.MolFromMolFile(out_sdf, sanitize=False) if success else None
            
            # =====================================================================
            # --- Stage 2: Explicit Solvent MD Evaluation ---
            # =====================================================================
            print(f"\n=== Stage 2: MD Execution ({method_name}) ===")
            md_avg, md_std = None, None
            md_prefix = os.path.join(base_dir, f"{method_name}_md")
            
            if MD_AVAILABLE and final_mol is not None:
                # Based on your openmm_md.py, it will output md_prefix.pdb and md_prefix.dcd
                # Using 25,000 steps for a quick test (approx 50ps). Adjust sim_steps as needed.
                md_avg, md_std, _ = run_short_md_evaluation(
                    rec_clean_pdb, final_mol, md_prefix, 
                    reference_mol=phenol_mol, sim_steps=2500000
                )
            
            md_rmsd_str = f"{md_avg:.2f}±{md_std:.2f}" if md_avg is not None else "N/A"
            print(f" -> End Stage 2: Smina ΔG={energy:.2f} | MD_RMSD={md_rmsd_str}")
            
            # =====================================================================
            # --- Stage 3: Standalone MM-GBSA (Decoupled) ---
            # =====================================================================
            expected_dcd = f"{md_prefix}.dcd"
            expected_pdb = f"{md_prefix}.pdb"
            mmgbsa_score = None
            
            if os.path.exists(expected_dcd) and os.path.exists(expected_pdb):
                print(f"\n=== Stage 3: MM-GBSA Analysis ({method_name}) ===")
                mmgbsa_score = calculate_mmgbsa_from_trajectory(
                    expected_pdb, expected_dcd, final_mol, base_dir, prefix=method_name
                )
            else:
                print("\n [Skip Stage 3] Trajectory or PDB not found. MD might have failed.")
            
            total_time = time.time() - t0_total
            
            results.append({
                "Method": method_name,
                "Smina_dG(kcal/mol)": energy,
                "MD_Avg_RMSD(A)": md_avg,
                "MD_RMSD_Std(A)": md_std,
                "Implicit_E(kcal/mol)": mmgbsa_score,
                "Time(s)": total_time
            })
                
        else:
            print(" [Graph] Failed to find clique.")

    print("\n" + "="*90)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    run_laccase_phenol_benchmark()