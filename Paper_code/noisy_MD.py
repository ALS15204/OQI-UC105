import os
import time
import subprocess
import requests
import shutil
import random
import pandas as pd
import numpy as np
import mdtraj as md
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

try:
    from src.openmm_md import run_short_md_evaluation
    MD_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] MD evaluation not available: {e}")
    MD_AVAILABLE = False

# =====================================================================
# --- 2. Local Helpers ---
# =====================================================================
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

def align_proteins(mobile_pdb_path, target_pdb_path, output_pdb_path):
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

def clean_receptor_pdb(input_pdb, output_pdb):
    """拔除受體中原有的配體 (如 MK1)，確保口袋是空的"""
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname not in ['HOH', 'WAT']:
                    continue
            f_out.write(line)

# =====================================================================
# --- 3. Standalone MM-GBSA Evaluator ---
# =====================================================================
def calculate_mmgbsa_from_trajectory(complex_pdb_path, dcd_path, ligand_mol, base_dir, prefix):
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
    traj = md.load_dcd(dcd_path, top=complex_pdb_path)
    selection_query = "(not water) and (not resname HOH) and (not resname NA) and (not resname CL) and (not element Na) and (not element Cl)"
    dry_indices = traj.topology.select(selection_query)
    
    if len(dry_indices) == 0: return None
        
    dry_traj = traj.atom_slice(dry_indices)
    dry_pdb_path = os.path.join(base_dir, f"{prefix}_dry.pdb")
    dry_traj[0].save_pdb(dry_pdb_path)
    
    print(" [MM-GBSA] Setting up Implicit Solvent System (Amber14 + igb)...")
    pdb = PDBFile(dry_pdb_path)
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
    for i in range(dry_traj.n_frames):
        positions = dry_traj.xyz[i] * 10.0 # nm to Angstroms
        context.setPositions(positions * unit.angstroms)
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        energies.append(pot_energy)
        
    avg_energy = np.mean(energies)
    std_energy = np.std(energies)
    print(f" [MM-GBSA] Average Complex Energy: {avg_energy:.2f} ± {std_energy:.2f} kcal/mol")
    return avg_energy

# =====================================================================
# --- 4. Main Experiment: Soft-Landing Noise Tolerance Benchmark ---
# =====================================================================
def run_1hxw_noise_benchmark():
    ligand_id = "1HSG"
    receptor_id = "1HXW" 
    
    print(f"=== Stage 1: Preparation | Cross-Docking ({ligand_id} -> {receptor_id}) ===")
    base_dir = "data_noise_1hxw"
    os.makedirs(base_dir, exist_ok=True)
    
    path_lig_complex = os.path.join(base_dir, f"{ligand_id}_complex.pdb")
    if not os.path.exists(path_lig_complex): download_pdb(ligand_id, path_lig_complex)
    
    path_rec_raw = os.path.join(base_dir, f"{receptor_id}_raw.pdb")
    if not os.path.exists(path_rec_raw): download_pdb(receptor_id, path_rec_raw)
        
    path_rec_empty = os.path.join(base_dir, f"{receptor_id}_empty.pdb")
    clean_receptor_pdb(path_rec_raw, path_rec_empty)

    path_rec_aligned = os.path.join(base_dir, f"{receptor_id}_aligned.pdb")
    align_proteins(path_rec_empty, path_lig_complex, path_rec_aligned)
    path_rec = path_rec_aligned if os.path.exists(path_rec_aligned) else path_rec_empty

    mol_lig_raw = Chem.MolFromPDBFile(path_lig_complex, removeHs=False, sanitize=True)
    ligand_mol, _ = extract_submol(mol_lig_raw)
    ligand_mol = AllChem.AddHs(ligand_mol, addCoords=True)
    
    pocket_center = get_mol_center(ligand_mol)
    print(f" [Setup] Pocket defined by: Reference_Ligand at {pocket_center}")

    rec_clean_pdb = os.path.join(base_dir, "clean_receptor.pdb")
    shutil.copy(path_rec, rec_clean_pdb)
    rec_qt = os.path.join(base_dir, "receptor.pdbqt")
    convert_to_pdbqt(path_rec, rec_qt)
    
    mol_rec_raw = Chem.MolFromPDBFile(path_rec, removeHs=False, sanitize=False)
    receptor_mol_pocket = extract_pocket_by_center(mol_rec_raw, pocket_center, cutoff=9.5)
    
    print("\n>>> Searching for the Ground Truth Clique...")
    G, nmap, lfeats, rfeats = build_pharmacophore_graph(ligand_mol, receptor_mol_pocket, tolerance=3)
    
    n = len(G.nodes())
    m = len(G.edges())
    rho = (2 * m) / (n * (n - 1)) if n > 1 else 0
    print(f"  Nodes (n)={n} | Density (rho)={rho:.3f}")
    if len(G.nodes()) == 0:
        print(" [Error] Generated graph is empty.")
        return

    solver = MaxCliqueSolver(G)
    res = solver.solve_sa(iterations=5000)
    perfect_clique = res['clique']
    
    print(f" -> Found Ground Truth Clique: Size {len(perfect_clique)}")
    
    # --- Simulating NISQ Fidelity ---
    experiments = [("Perfect_100pct", perfect_clique)]
    
    if len(perfect_clique) >= 10:
        drops = [int(len(perfect_clique) * r) for r in (0.1, 0.2, 0.3, 0.4)]
        labels = ["Noisy_90pct", "Noisy_80pct", "Noisy_70pct", "Noisy_60pct"]
        
        for drop, label in zip(drops, labels):
            noisy_c = perfect_clique.copy()
            for _ in range(drop): noisy_c.pop(random.randint(0, len(noisy_c)-1))
            experiments.append((label, noisy_c))
    else:
        print(" [Warning] Clique too small for percentage drop. Running 100% only.")

    results = []
    
    for exp_name, test_clique in experiments:
        print(f"\n{'='*50}")
        print(f" Testing Quantum Output: {exp_name} (Size: {len(test_clique)})")
        print(f"{'='*50}")
        t0_total = time.time()
        
        docked_mol = generate_docked_pdb(test_clique, nmap, lfeats, rfeats, ligand_mol, 
                                         os.path.join(base_dir, f"pose_{exp_name}.pdb"))
        
        print(" -> Running Smina Local Refinement...")
        docked_mol_safe = enforce_physical_distance(docked_mol, pocket_center, is_metal=False)
        in_sdf = os.path.join(base_dir, f"{exp_name}_in.sdf")
        out_sdf = os.path.join(base_dir, f"{exp_name}_out.sdf")
        prepare_ligand_for_smina(docked_mol_safe, in_sdf)
        
        energy, success = run_smina_sdf(
            rec_qt, in_sdf, out_sdf, 
            center=get_mol_center(docked_mol_safe), 
            size=(10, 10, 10),
            extra_args="--exhaustiveness 16" 
        )
        final_mol = Chem.MolFromMolFile(out_sdf, sanitize=False) if success else docked_mol_safe
        print(f"    [Smina ΔG] = {energy:.2f} kcal/mol")
        
        print(f"\n -> Running Explicit Solvent MD ({exp_name})...")
        md_avg, md_std = None, None
        md_prefix = os.path.join(base_dir, f"{exp_name}_md")
        
        if MD_AVAILABLE and final_mol is not None:
            try:
                md_avg, md_std, _ = run_short_md_evaluation(
                    rec_clean_pdb, final_mol, md_prefix, 
                    reference_mol=ligand_mol, sim_steps=50000 
                )
            except Exception as e:
                print(f"    [MD Error] Simulation exploded: {e}")
        
        expected_dcd = f"{md_prefix}.dcd"
        expected_pdb = f"{md_prefix}.pdb"
        mmgbsa_score = None
        
        if os.path.exists(expected_dcd) and os.path.exists(expected_pdb) and (md_avg is not None):
            print(f"\n -> Running MM-GBSA Analysis ({exp_name})...")
            try:
                mmgbsa_score = calculate_mmgbsa_from_trajectory(
                    expected_pdb, expected_dcd, final_mol, base_dir, prefix=exp_name
                )
            except Exception as e:
                print(f"    [MM-GBSA Error] Energy calculation failed: {e}")
        
        results.append({
            "Quantum_Output": exp_name,
            "Clique_Size": len(test_clique),
            "Smina_dG": energy,
            "MD_Avg_RMSD(A)": md_avg,
            "MD_RMSD_Std(A)": md_std,
            "Implicit_E(kcal/mol)": mmgbsa_score,
            "Time(s)": time.time() - t0_total
        })

    print("\n" + "="*85)
    print(" COMPLEX REGIME: NOISE-TOLERANCE BENCHMARK REPORT (1HSG -> 1HXW)")
    print("="*85)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*85)


if __name__ == "__main__":
    run_1hxw_noise_benchmark()