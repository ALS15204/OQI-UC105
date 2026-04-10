import os
import numpy as np
import networkx as nx
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

# =============================================================================
# 1. ROBUST MOLECULE EXTRACTION
# =============================================================================
def extract_submol(complex_mol):
    """
    Separates Ligand and Receptor robustly.
    Filters out Protein and known Artifacts. Everything else is Ligand.
    """
    if complex_mol is None: return Chem.Mol(), Chem.Mol()

    ligand_atoms = []
    receptor_atoms = []
    
    # 1. Protein Residues (Keep in Receptor)
    protein_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "ASX", "GLX", "UNK"
    }
    
    # 2. Artifacts (Discard)
    artifacts = {
        "HOH", "WAT", "TIP", "SOL",             # Water
        "NA", "CL", "CA", "K", "SO4", "PO4",    # Common Ions/Salts
        "PEG", "BOG", "DMS", "EDO", "GOL", "ACT" # Cryoprotectants/Buffers
    }
    
    # 3. Metals (Keep in Receptor for coordination graph)
    metals = {"ZN", "CU", "FE", "MN", "MG", "CO", "NI", "HG", "CD"}

    for atom in complex_mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if not res: continue
        res_name = res.GetResidueName().strip().upper()
        
        if res_name in protein_residues:
            receptor_atoms.append(atom.GetIdx())
        elif res_name in metals:
            receptor_atoms.append(atom.GetIdx())
        elif res_name in artifacts:
            pass # Discard junk
        else:
            # Not protein, not junk -> Likely the Ligand
            ligand_atoms.append(atom.GetIdx())

    def _create_mol(indices):
        if not indices: return None
        rw = Chem.RWMol(complex_mol)
        keep = set(indices)
        all_idxs = set(range(complex_mol.GetNumAtoms()))
        for i in sorted(list(all_idxs - keep), reverse=True): rw.RemoveAtom(i)
        try:
            m = rw.GetMol()
            try: 
                m.UpdatePropertyCache(strict=False)
                # [FIX] Force ring info initialization for receptor fragments
                Chem.GetSSSR(m) 
            except: pass
            return m
        except: return None

    # Create Mols
    rec_mol = _create_mol(receptor_atoms)
    if rec_mol is None: rec_mol = Chem.Mol()

    raw_ligand = _create_mol(ligand_atoms)
    final_ligand = Chem.Mol()
    
    # Filter Ligand: Keep largest organic fragment
    if raw_ligand and raw_ligand.GetNumAtoms() > 0:
        frags = Chem.GetMolFrags(raw_ligand, asMols=True, sanitizeFrags=False)
        if frags:
            best_frag = max(frags, key=lambda m: m.GetNumAtoms())
            final_ligand = best_frag
            try: Chem.SanitizeMol(final_ligand)
            except: pass

    return final_ligand, rec_mol

# =============================================================================
# 2. POCKET DETECTION & MANIPULATION
# =============================================================================
def get_pocket_center_generic(receptor_mol, target_metal_symbol=None):
    """
    Finds the binding pocket center.
    Priority 1: Target Metal (if specified and found).
    Priority 2: Original Ligand (HETATMs in receptor).
    Priority 3: Geometric Center (Fallback).
    Returns: (coords_array, source_type_string)
    """
    # 1. Try Metal
    if target_metal_symbol:
        for atom in receptor_mol.GetAtoms():
            if atom.GetSymbol().upper() == target_metal_symbol.upper():
                pos = receptor_mol.GetConformer().GetAtomPosition(atom.GetIdx())
                return np.array([pos.x, pos.y, pos.z]), f"Metal_{target_metal_symbol}"

    # 2. Try Original Ligand (Scan for HETATMs that are not water/ions)
    junk = {"HOH", "WAT", "SOL", "NA", "CL", "SO4", "PO4"}
    het_coords = []
    
    for atom in receptor_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetIsHeteroAtom():
            if info.GetResidueName().strip().upper() not in junk:
                pos = receptor_mol.GetConformer().GetAtomPosition(atom.GetIdx())
                het_coords.append([pos.x, pos.y, pos.z])
    
    if het_coords:
        return np.mean(het_coords, axis=0), "Original_Ligand"

    # 3. Fallback
    conf = receptor_mol.GetConformer()
    pts = [conf.GetAtomPosition(i) for i in range(receptor_mol.GetNumAtoms())]
    if pts:
        return np.mean(pts, axis=0), "Protein_Center_Fallback"
    
    return np.array([0,0,0]), "Origin_Error"

def extract_pocket_by_center(receptor_mol, center, cutoff=18.0):
    """Slices receptor atoms around a 3D point."""
    if receptor_mol is None: return receptor_mol
    
    indices = []
    conf = receptor_mol.GetConformer()
    for atom in receptor_mol.GetAtoms():
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        if np.linalg.norm(pos - center) <= cutoff:
            indices.append(atom.GetIdx())
            
    if not indices: return receptor_mol
    
    new_mol = Chem.RWMol(receptor_mol)
    keep = set(indices)
    all_idx = set(range(receptor_mol.GetNumAtoms()))
    for i in sorted(list(all_idx - keep), reverse=True):
        new_mol.RemoveAtom(i)
        
    mol = new_mol.GetMol()
    
    # [CRITICAL FIX] Initialize Ring Information for the sliced pocket
    # Without this, FeatureFactory crashes on "RingInfo not initialized"
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol) 
    except: pass
    
    return mol

def enforce_physical_distance(ligand_mol, target_center, is_metal=False, safe_dist=2.8):
    """
    Smart Enforcer:
    - If Target is Metal: Push atoms away to safe_dist (Vina hates metal overlap).
    - If Target is Pocket: Do NOTHING (We want to be inside the pocket!).
    """
    if not is_metal: 
        return ligand_mol # Pass through for non-metal pockets
    
    # Metal Logic (Push away)
    conf = ligand_mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
    dists = np.linalg.norm(positions - target_center, axis=1)
    
    if np.min(dists) < safe_dist:
        print(f"   [Enforcer] Clash detected ({np.min(dists):.2f} A). Pushing away from Metal...")
        # Simple push: move whole molecule along the vector from metal to ligand center
        lig_center = np.mean(positions, axis=0)
        vec = lig_center - target_center
        norm = np.linalg.norm(vec)
        if norm < 0.1: vec = np.array([1.0, 0, 0]) # Random escape
        else: vec /= norm
        
        push_vec = vec * (safe_dist - np.min(dists) + 0.2)
        new_pos = positions + push_vec
        
        new_mol = Chem.Mol(ligand_mol)
        new_conf = new_mol.GetConformer()
        for i in range(new_mol.GetNumAtoms()):
            new_conf.SetAtomPosition(i, list(new_pos[i]))
        return new_mol
        
    return ligand_mol

# =============================================================================
# 3. GRAPH CONSTRUCTION
# =============================================================================

def build_quantum_pharmacophore_graph(ligand_mol, receptor_mol, max_tolerance=3.5, alpha=1.0, reward=-2.0, penalty=10.0):
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    
    receptor_mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(receptor_mol)
    
    lig_feats = [{'f': f.GetFamily(), 'c': np.array(f.GetPos())} for f in factory.GetFeaturesForMol(ligand_mol)]
    rec_feats = [{'f': f.GetFamily(), 'c': np.array(f.GetPos())} for f in factory.GetFeaturesForMol(receptor_mol)]
    
    # Add Metal Features manually
    target_metals = ["ZN", "CU", "FE", "MN", "MG", "CO", "NI"]
    conf = receptor_mol.GetConformer()
    for atom in receptor_mol.GetAtoms():
        if atom.GetSymbol().upper() in target_metals:
            pos = conf.GetAtomPosition(atom.GetIdx())
            rec_feats.append({'f': 'Metal', 'c': np.array([pos.x, pos.y, pos.z])})

    G = nx.Graph()
    Q_dict = {} 
    node_map = {}
    idx = 0
    valid = {('Donor','Acceptor'), ('Acceptor','Donor'), ('Aromatic','Aromatic'), 
             ('Hydrophobe','Hydrophobe'), ('PosIonizable','NegIonizable'), ('NegIonizable','PosIonizable'),
             ('Donor','Metal'), ('Acceptor','Metal'), ('NegIonizable','Metal'), ('Aromatic','Metal')}

    for i, lf in enumerate(lig_feats):
        for j, rf in enumerate(rec_feats):
            if (lf['f'], rf['f']) in valid:
                G.add_node(idx, l=i, r=j)
                node_map[idx] = (i, j)
                Q_dict[(idx, idx)] = reward
                idx += 1
                
    nodes = list(G.nodes())
    
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            l1, r1 = node_map[u]
            l2, r2 = node_map[v]

            if l1 == l2 or r1 == r2:
                Q_dict[(u, v)] = penalty
                continue
                
            d1 = np.linalg.norm(lig_feats[l1]['c'] - lig_feats[l2]['c'])
            d2 = np.linalg.norm(rec_feats[r1]['c'] - rec_feats[r2]['c'])
            diff = abs(d1 - d2)
            
            if diff <= max_tolerance:
                edge_weight = alpha * (diff ** 2)
                
                G.add_edge(u, v, weight=edge_weight)
                
                Q_dict[(u, v)] = edge_weight
            else:
                Q_dict[(u, v)] = penalty
                
    return G, node_map, lig_feats, rec_feats, Q_dict

def build_pharmacophore_graph(ligand_mol, receptor_mol, tolerance=1.5):
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    
    lig_feats = [{'f': f.GetFamily(), 'c': np.array(f.GetPos())} for f in factory.GetFeaturesForMol(ligand_mol)]
    rec_feats = [{'f': f.GetFamily(), 'c': np.array(f.GetPos())} for f in factory.GetFeaturesForMol(receptor_mol)]
    
    # Add Metal Features manually
    target_metals = ["ZN", "CU", "FE", "MN", "MG", "CO", "NI"]
    conf = receptor_mol.GetConformer()
    for atom in receptor_mol.GetAtoms():
        if atom.GetSymbol().upper() in target_metals:
            pos = conf.GetAtomPosition(atom.GetIdx())
            rec_feats.append({'f': 'Metal', 'c': np.array([pos.x, pos.y, pos.z])})

    G = nx.Graph()
    node_map = {}
    idx = 0
    valid = {('Donor','Acceptor'), ('Acceptor','Donor'), ('Aromatic','Aromatic'), 
             ('Hydrophobe','Hydrophobe'), ('PosIonizable','NegIonizable'), ('NegIonizable','PosIonizable'),
             ('Donor','Metal'), ('Acceptor','Metal'), ('NegIonizable','Metal'), ('Aromatic','Metal')}

    for i, lf in enumerate(lig_feats):
        for j, rf in enumerate(rec_feats):
            if (lf['f'], rf['f']) in valid:
                G.add_node(idx, l=i, r=j)
                node_map[idx] = (i, j)
                idx += 1
                
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            l1, r1 = node_map[u]
            l2, r2 = node_map[v]
            d1 = np.linalg.norm(lig_feats[l1]['c'] - lig_feats[l2]['c'])
            d2 = np.linalg.norm(rec_feats[r1]['c'] - rec_feats[r2]['c'])
            if abs(d1 - d2) < tolerance:
                G.add_edge(u, v)
                
    return G, node_map, lig_feats, rec_feats