import os
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# OpenMM Imports
try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from openmmforcefields.generators import GAFFTemplateGenerator  # Use GAFF for maximum robustness
except ImportError:
    pass

def clean_protein_for_openmm(input_pdb, output_pdb):
    """Remove waters and non-standard residues to avoid conflicts in implicit solvent."""
    keep_res = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN"  # Explicitly exclude HOH
    }
    with open(input_pdb, 'r') as fin, open(output_pdb, 'w') as fout:
        for line in fin:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                if res_name in keep_res:
                    fout.write(line)
            elif line.startswith("TER") or line.startswith("END"):
                fout.write(line)
    return output_pdb

def run_openmm_refinement(receptor_pdb, ligand_sdf, reference_mol=None):
    """
    OpenMM refinement using GAFF (highly robust for docked poses).
    reference_mol (crystal ligand) is REQUIRED to restore bond orders/aromaticity destroyed by Smina.
    """
    t0 = time.time()
    base_dir = os.path.dirname(ligand_sdf)
    clean_rec = os.path.join(base_dir, "temp_openmm_rec.pdb")
    clean_protein_for_openmm(receptor_pdb, clean_rec)
    
    try:
        # 1. Load Protein
        pdb = app.PDBFile(clean_rec)
        print(f"   [OpenMM] Protein loaded: {pdb.topology.getNumAtoms()} atoms")
        
        # 2. Load Ligand from Smina SDF and FIX bond orders
        print(f"   [OpenMM] Loading ligand from SDF and repairing bond orders...")
        suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False, sanitize=False)
        docked_raw = suppl[0] if len(suppl) > 0 else None
        
        if docked_raw is None:
            print("   [OpenMM Error] Cannot read ligand SDF.")
            return None, 0.0, None

        # Critical repair using crystal ligand template
        if reference_mol is not None:
            try:
                ref_heavy = Chem.RemoveHs(reference_mol)
                docked_heavy = Chem.RemoveHs(docked_raw)
                fixed_mol = AllChem.AssignBondOrdersFromTemplate(ref_heavy, docked_heavy)
                fixed_mol = Chem.AddHs(fixed_mol, addCoords=True)
                print("   [OpenMM] Bond orders successfully restored using template.")
            except Exception as e:
                print(f"   [OpenMM Warning] Template repair failed ({e}), using raw docked mol.")
                fixed_mol = Chem.AddHs(docked_raw, addCoords=True)
        else:
            print("   [OpenMM Warning] No reference_mol provided, using raw docked mol (risky).")
            fixed_mol = Chem.AddHs(docked_raw, addCoords=True)

        # 3. Parameterize with GAFF (very tolerant)
        forcefield = app.ForceField('amber14/protein.ff14SB.xml')
        
        gaff = GAFFTemplateGenerator(molecules=fixed_mol, version='2.11')  # GAFF 2.11
        forcefield.registerTemplateGenerator(gaff.generator)
        print("   [OpenMM] GAFF templates registered.")

        # 4. Build combined topology (protein + ligand)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        
        # Add ligand topology and positions (use RDKit conformer directly)
        lig_topology = fixed_mol.ToTopology() if hasattr(fixed_mol, 'ToTopology') else None
        # Fallback for older RDKit: create topology manually
        if lig_topology is None:
            from rdkit.Chem.rdchem import EditableMol
            # Simple way: use Modeller to add atoms one by one (robust)
            modeller.add(fixed_mol, fixed_mol.GetConformer().GetPositions())

        modeller.add(fixed_mol, fixed_mol.GetConformer().GetPositions())  # This works in most cases

        print("   [OpenMM] Combined system topology built.")

        # 5. Create System (implicit solvent)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=app.OBC2,
            implicitSolventSaltConc=0.15*unit.molar,  # Physiological salt
            solventDielectric=78.5,
            soluteDielectric=1
        )
        print("   [OpenMM] System created successfully.")

        # 6. Minimization
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        
        print(f"   [OpenMM] Starting minimization (max 3000 iterations)...")
        simulation.minimizeEnergy(maxIterations=3000, tolerance=1.0*unit.kilojoule_per_mole)
        
        # 7. Extract refined ligand positions and energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)
        
        n_prot = pdb.topology.getNumAtoms()
        lig_pos = positions[n_prot:]
        
        print(f"   [OpenMM] Refinement completed. Final energy: {energy:.2f} kcal/mol")
        return energy, time.time() - t0, lig_pos

    except Exception as e:
        print(f"   [OpenMM Error] {e}")
        import traceback
        traceback.print_exc()  # Full traceback for precise debugging
        return None, 0.0, None