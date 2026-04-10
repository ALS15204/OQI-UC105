import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import openmm.app as app
import openmm as mm
from openmm import unit
import time
from rdkit.Chem import rdMolAlign, AllChem

def calculate_rmsd(ref_mol, target_mol, mode="best"):
    """
    Calculates RMSD in different modes.
    """
    try:
        # Handle atom count mismatch by removing H if needed
        if ref_mol.GetNumAtoms() != target_mol.GetNumAtoms():
            ref_no_h = Chem.RemoveHs(ref_mol)
            target_no_h = Chem.RemoveHs(target_mol)
            if ref_no_h.GetNumAtoms() == target_no_h.GetNumAtoms():
                return calculate_rmsd(ref_no_h, target_no_h, mode=mode)
            print(f" [RMSD] Atom count mismatch. Returning 999.0")
            return 999.0
        
        # Coordinate transfer to ensure identical topology
        pose_with_ref_topology = Chem.Mol(ref_mol)
        target_conf = target_mol.GetConformer()
        new_conf = pose_with_ref_topology.GetConformer()
        for i in range(pose_with_ref_topology.GetNumAtoms()):
            pos = target_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        
        if mode == "best":
            return rdMolAlign.GetBestRMS(pose_with_ref_topology, ref_mol)
        
        elif mode == "heavy":
            # Heavy-atom only (most common standard)
            ref_heavy = Chem.RemoveHs(ref_mol)
            pose_heavy = Chem.RemoveHs(pose_with_ref_topology)
            return rdMolAlign.GetBestRMS(pose_heavy, ref_heavy)
        
        else:
            return rdMolAlign.GetBestRMS(pose_with_ref_topology, ref_mol)
    
    except Exception as e:
        print(f" [RMSD] Error: {e}")
        return 999.0

def calculate_mmff_energy(mol):
    """
    Calculates MMFF94 energy (Internal Strain).
    Lower is better. High values (>100) mean bad geometry.
    """
    if mol is None: return 999.0
    try:
        # Clone to avoid modifying original
        mol_calc = Chem.Mol(mol)
        try: Chem.SanitizeMol(mol_calc)
        except: pass
        
        mol_calc = Chem.AddHs(mol_calc, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol_calc)
        
        if props is None: return 999.0 # MMFF parameterization failed
        
        ff = AllChem.MMFFGetMoleculeForceField(mol_calc, props)
        if ff:
            return ff.CalcEnergy()
    except Exception:
        return 999.0
    return 999.0

def run_amber_minimization(prmtop_path, inpcrd_path, gb_model="OBC2", max_iter=200, restrain_ligand=True):
    """
    Run energy minimization using OpenMM.
    Fixed restraint to non-periodic formula (avoid periodicdistance bug in implicit solvent).
    Restrain protein backbone (CA) and ligand heavy atoms to prevent Zn/Cu repulsion drift.
    """
    try:
        prmtop = app.AmberPrmtopFile(prmtop_path)
        inpcrd = app.AmberInpcrdFile(inpcrd_path)
       
        system = prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=app.OBC2
        )
       
        # Diagnostic: Initial energy before restraint
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtoseconds)
        simulation = app.Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
        init_state = simulation.context.getState(getEnergy=True)
        init_energy = init_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        print(f" [MM] Initial Energy (pre-restraint): {init_energy:.2f} kcal/mol")
       
        if restrain_ligand:
            # Correct non-periodic restraint: distance squared
            restraint = mm.CustomExternalForce('k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
            restraint.addGlobalParameter('k', 50.0 * unit.kilocalories_per_mole / unit.angstroms**2)  # Moderate
            restraint.addPerParticleParameter('x0')
            restraint.addPerParticleParameter('y0')
            restraint.addPerParticleParameter('z0')
           
            positions = inpcrd.positions
            total_atoms = prmtop.topology.getNumAtoms()
            # Robust ligand detection: assume ligand is last ~100 atoms (adjust if large ligand)
            ligand_start_idx = max(0, total_atoms - 100)  # Safe fallback
           
            for i, atom in enumerate(prmtop.topology.atoms()):
                restrain = False
                # Protein CA
                if atom.name == 'CA' and atom.residue.name not in ['LIG', 'UNL', 'MOL']:
                    restrain = True
                # Ligand heavy atoms (last atoms in topology)
                if i >= ligand_start_idx and atom.element.symbol != 'H':
                    restrain = True
               
                if restrain:
                    pos = positions[i].value_in_unit(unit.nanometers)
                    restraint.addParticle(i, pos)
           
            system.addForce(restraint)
            print(f" [MM] Applied restraints (protein CA + ligand heavy atoms).")
       
        # Minimization
        t0 = time.time()
        simulation.minimizeEnergy(maxIterations=max_iter)
        t1 = time.time()
       
        final_state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = final_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        pos_ang = final_state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)
        print(f" [MM] Final Energy: {energy:.2f} kcal/mol")
       
        return energy, (t1 - t0), pos_ang
       
    except Exception as e:
        print(f" [OpenMM Error] {str(e)}")
        return None, 0.0, None