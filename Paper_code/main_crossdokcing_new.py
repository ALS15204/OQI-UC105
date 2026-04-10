import os
import time
import subprocess
import requests
import shutil
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

# --- 1. Import Custom Modules ---
from src.solvers import MaxCliqueSolver
from src.mol_prep import (
    extract_submol, 
    get_pocket_center_generic, 
    extract_pocket_by_center, 
    build_pharmacophore_graph, 
    enforce_physical_distance
)

# --- MD evaluation module ---
try:
    from src.openmm_md import run_short_md_evaluation
    MD_AVAILABLE = True
    print("[Info] MD evaluation module loaded successfully.")
except ImportError as e:
    print(f"[Warning] MD evaluation not available: {e}. Will skip MD stability analysis.")
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
    
    # Kabsch Algorithm
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
    
    for atom in new_mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("UNL")
        info.SetIsHeteroAtom(True)      
        atom.SetMonomerInfo(info)

    Chem.MolToPDBFile(new_mol, output_path)
    return new_mol


def prepare_ligand_for_smina(mol, output_path):
    w = Chem.SDWriter(output_path)
    w.write(mol)
    w.close()
    return True

def calculate_rmsd_robust(ref_mol, docked_mol):
    if ref_mol is None or docked_mol is None: return 999.0
    try:
        ref_heavy = Chem.RemoveHs(ref_mol)
        dock_heavy = Chem.RemoveHs(docked_mol)
        if ref_heavy.GetNumAtoms() != dock_heavy.GetNumAtoms(): return 999.0
        try: 
            return Chem.rdMolAlign.GetBestRMS(ref_heavy, dock_heavy)
        except:
            ref_conf = ref_heavy.GetConformer()
            dock_conf = dock_heavy.GetConformer()
            c1 = np.array([ref_conf.GetAtomPosition(i) for i in range(ref_heavy.GetNumAtoms())])
            c2 = np.array([dock_conf.GetAtomPosition(i) for i in range(dock_heavy.GetNumAtoms())])
            return np.sqrt(np.sum((c1-c2)**2)/len(c1))
    except: 
        return 999.0

def align_proteins(mobile_pdb_path, target_pdb_path, output_pdb_path):
    # 用你原本的 Biopython 對齊，因為 1HSG 和 1HXW 序列完全一樣，這個絕對不會出錯
    try:
        from Bio.PDB import PDBParser, Superimposer, PDBIO, PPBuilder
        from Bio import Align
    except ImportError:
        print(" [Error] Biopython is required.")
        return False

    print(f"   [Align] Superimposing {os.path.basename(mobile_pdb_path)} onto {os.path.basename(target_pdb_path)}...")
    parser = PDBParser(QUIET=True)
    mob_st = parser.get_structure('mobile', mobile_pdb_path)
    tgt_st = parser.get_structure('target', target_pdb_path)

    ppb = PPBuilder()
    mob_peps = ppb.build_peptides(mob_st[0])
    tgt_peps = ppb.build_peptides(tgt_st[0])

    mob_pep = max(mob_peps, key=lambda p: len(p.get_sequence()))
    tgt_pep = max(tgt_peps, key=lambda p: len(p.get_sequence()))

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    best_aln = aligner.align(str(tgt_pep.get_sequence()), str(mob_pep.get_sequence()))[0]

    tgt_atoms, mob_atoms = [], []
    for (t_start, t_end), (m_start, m_end) in zip(best_aln.aligned[0], best_aln.aligned[1]):
        for i in range(t_end - t_start):
            res_t = tgt_pep[t_start + i]
            res_m = mob_pep[m_start + i]
            if res_t.has_id('CA') and res_m.has_id('CA'):
                tgt_atoms.append(res_t['CA'])
                mob_atoms.append(res_m['CA'])

    sup = Superimposer()
    sup.set_atoms(tgt_atoms, mob_atoms)
    sup.apply(mob_st[0].get_atoms())

    io = PDBIO()
    io.set_structure(mob_st)
    io.save(output_pdb_path)
    return True

# --- 解決問題的關鍵：把原生配體 MK1 拔掉 ---
def clean_receptor_pdb(input_pdb, output_pdb):
    """拔除受體中原有的配體 (如 MK1)，確保口袋是空的"""
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                # 只保留水分子，拔掉所有的外來配體
                if resname not in ['HOH', 'WAT']:
                    continue
            f_out.write(line)

def run_smina_sdf(receptor_path, ligand_path, output_path, center=None, size=(20, 20, 20), extra_args=""):
    cmd = ["smina", "-r", receptor_path, "-l", ligand_path, "-o", output_path, "--cpu", "4", "--quiet"]
    if center is not None:
        cmd.extend(["--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
                    "--size_x", f"{size[0]}", "--size_y", f"{size[1]}", "--size_z", f"{size[2]}"])
    else:
        cmd.append("--autobox_ligand")
        cmd.append(ligand_path)
        
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
    except:
        return 0.0, False

def run_pipeline(method_name, ligand_mol, rec_qt, rec_clean_pdb, base_dir, docked_mol_input, pocket_center, is_metal_pocket):
    if docked_mol_input is None: return 999.0, 0.0, None, None, None, None
    
    docked_mol_safe = enforce_physical_distance(docked_mol_input, pocket_center, is_metal=is_metal_pocket)
    in_sdf = os.path.join(base_dir, f"{method_name}_in.sdf")
    out_sdf = os.path.join(base_dir, f"{method_name}_out.sdf")
    prepare_ligand_for_smina(docked_mol_safe, in_sdf)
    
    current_center = get_mol_center(docked_mol_safe)
    energy, success = run_smina_sdf(rec_qt, in_sdf, out_sdf, center=current_center, size=(10, 10, 10), extra_args="--exhaustiveness 8")
    
    rmsd = 999.0
    final_mol = None
    if success:
        final_mol = Chem.MolFromMolFile(out_sdf, sanitize=False)
        rmsd = calculate_rmsd_robust(ligand_mol, final_mol)
    
    md_avg, md_std, md_energy = None, None, None
    if MD_AVAILABLE and final_mol is not None:
        md_prefix = os.path.join(base_dir, f"{method_name}_md")
        md_avg, md_std, md_energy = run_short_md_evaluation(rec_clean_pdb, final_mol, md_prefix, reference_mol=ligand_mol, sim_steps=50000)
    
    return rmsd, energy, md_avg, md_std, md_energy, final_mol

# --- 6. Main Benchmark ---
def run_cross_docking_benchmark():
    ligand_id, receptor_id = "1HSG", "1HXW" 
    
    print(f"=== Cross-Docking Benchmark: {ligand_id} (Lig) -> {receptor_id} (Rec) ===")
    base_dir = "data_cross"
    os.makedirs(base_dir, exist_ok=True)
    
    path_lig_complex = os.path.join(base_dir, f"{ligand_id}_complex.pdb")
    if not os.path.exists(path_lig_complex): download_pdb(ligand_id, path_lig_complex)
    
    path_rec_raw = os.path.join(base_dir, f"{receptor_id}_raw.pdb")
    if not os.path.exists(path_rec_raw): download_pdb(receptor_id, path_rec_raw)

    # ==========================================
    # 這裡就是關鍵：把 1HXW 裡面的 MK1 拔掉！
    # ==========================================
    path_rec_empty = os.path.join(base_dir, f"{receptor_id}_empty.pdb")
    clean_receptor_pdb(path_rec_raw, path_rec_empty)

    path_rec_aligned = os.path.join(base_dir, f"{receptor_id}_aligned.pdb")
    align_proteins(path_rec_empty, path_lig_complex, path_rec_aligned)
    path_rec = path_rec_aligned if os.path.exists(path_rec_aligned) else path_rec_empty

    mol_lig_raw = Chem.MolFromPDBFile(path_lig_complex, removeHs=False, sanitize=True)
    ligand_mol, _ = extract_submol(mol_lig_raw)
    ligand_mol = AllChem.AddHs(ligand_mol, addCoords=True)
    
    pocket_center = get_mol_center(ligand_mol)
    is_metal_pocket = False 
    print(f" [Setup] Pocket defined by: Reference_Ligand at {pocket_center}")
    
    # MD 用乾淨的 (沒有MK1的) PDB
    rec_clean_pdb = os.path.join(base_dir, "clean_receptor.pdb")
    shutil.copy(path_rec, rec_clean_pdb)
    
    # Smina 也是用乾淨的 (沒有MK1的) 蛋白質轉 PDBQT
    rec_qt = os.path.join(base_dir, "receptor.pdbqt")
    convert_to_pdbqt(path_rec, rec_qt)
    
    mol_rec_raw = Chem.MolFromPDBFile(path_rec, removeHs=False, sanitize=False)
    receptor_mol_pocket = extract_pocket_by_center(mol_rec_raw, pocket_center, cutoff=20.0)
    
    results = []
    methods = ["Graph_Greedy", "Graph_SA"]
    tolerances = [1.5, 2.0]

    for method_name in methods:
        print(f"\n>>> Method: {method_name}")
        t0_total = time.time()
        
        best_clique = []
        best_data = None
        
        for tol in tolerances:
            G, nmap, lfeats, rfeats = build_pharmacophore_graph(ligand_mol, receptor_mol_pocket, tolerance=tol)
            if len(G.nodes()) == 0: continue
            
            solver = MaxCliqueSolver(G)
            res = solver.solve_sa(node_weights=None, iterations=2000) if method_name == "Graph_SA" else solver.solve_greedy(node_weights=None)
            
            if len(res['clique']) > len(best_clique):
                best_clique = res['clique']
                best_data = (nmap, lfeats, rfeats)
                if len(best_clique) >= 15: break
        
        if best_clique:
            print(f" [Graph] Best Clique Size: {len(best_clique)}")
            nmap, lfeats, rfeats = best_data
            docked_mol = generate_docked_pdb(best_clique, nmap, lfeats, rfeats, ligand_mol, os.path.join(base_dir, f"pose_{method_name}.pdb"))
            
            rmsd, energy, md_avg, md_std, md_energy, _ = run_pipeline(
                method_name, ligand_mol, rec_qt, rec_clean_pdb, base_dir, docked_mol, pocket_center, is_metal_pocket
            )
            
            md_rmsd_str = f"{md_avg:.2f}±{md_std:.2f}" if md_avg is not None else "N/A"
            total_time = time.time() - t0_total
            print(f" -> Result: RMSD={rmsd:.2f} Å | Affinity={energy:.2f} kcal/mol | MD_RMSD={md_rmsd_str} Å | Time={total_time:.2f}s")
            
            results.append({"Method": method_name, "RMSD(A)": rmsd, "Affinity": energy, "MD_Avg_RMSD(A)": md_avg if md_avg is not None else "N/A", "MD_Std": md_std if md_std is not None else "N/A", "Time(s)": total_time})
        else:
            print(" [Graph] Failed to find clique.")

    print("\n>>> Method: Pure_Smina")
    t0 = time.time()
    out_global = os.path.join(base_dir, "global_out.sdf")
    in_global = os.path.join(base_dir, "global_in.sdf")
    prepare_ligand_for_smina(ligand_mol, in_global)
    
    score_g, success = run_smina_sdf(rec_qt, in_global, out_global, center=pocket_center, size=(25,25,25), extra_args="--exhaustiveness 16")
    
    rmsd_g = 999.0
    mol_global = None
    if success:
        mol_global = Chem.MolFromMolFile(out_global, sanitize=False)
        rmsd_g = calculate_rmsd_robust(ligand_mol, mol_global)
    
    md_avg_g, md_std_g, md_energy_g = None, None, None
    if MD_AVAILABLE and mol_global is not None:
        try:
            md_avg_g, md_std_g, md_energy_g = run_short_md_evaluation(rec_clean_pdb, mol_global, os.path.join(base_dir, "Pure_Smina_md"), reference_mol=ligand_mol, sim_steps=50000)
        except: pass
    
    md_rmsd_str_g = f"{md_avg_g:.2f}±{md_std_g:.2f}" if md_avg_g is not None else "N/A"
    t_global = time.time() - t0
    print(f" -> Result: RMSD={rmsd_g:.2f} Å | Affinity={score_g:.2f} kcal/mol | MD_RMSD={md_rmsd_str_g} Å | Time={t_global:.2f}s")
    
    results.append({"Method": "Pure_Smina", "RMSD(A)": rmsd_g, "Affinity": score_g, "MD_Avg_RMSD(A)": md_avg_g if md_avg_g is not None else "N/A", "MD_Std": md_std_g if md_std_g is not None else "N/A", "Time(s)": t_global})
    
    print("\n" + "="*85)
    print(" BASELINE BENCHMARK REPORT (1HSG -> 1HXW)")
    print("="*85)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    run_cross_docking_benchmark()