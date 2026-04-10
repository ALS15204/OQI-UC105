# src/openmm_md.py
# Final Golden Version
# Features: Explicit Solvent (TIP3P) + CUDA Acceleration + Topology-Based Analysis
# Logic:
# 1. Loads Plugins (Fixes Environment issues)
# 2. Adds Water Box (Fixes RMSD Inf & XML Errors)
# 3. Uses CUDA (Fixes Slowness)
# 4. Index/Topology-Based Analysis (Fixes RMSD N/A by identifying ligand as the last residue)

import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Plugin Auto-Loader ---
# Automatically finds and loads OpenMM plugins (e.g., CUDA, CPU) 
# if the environment paths are messy.
def fix_openmm_plugins():
    import openmm as mm
    paths = [
        os.path.join(sys.prefix, 'lib', 'plugins'),
        os.path.join(sys.prefix, 'Library', 'lib', 'plugins'),
    ]
    for p in paths:
        if os.path.exists(p) and os.path.isdir(p):
            try:
                # Load quietly
                mm.Platform.loadPluginsFromDirectory(p)
            except:
                pass
# ---------------------------

try:
    import openmm as mm
    fix_openmm_plugins() 
    import openmm.app as app
    from openmm import unit
    from openmmforcefields.generators import GAFFTemplateGenerator
    from openff.toolkit.topology import Molecule
    import mdtraj as md
    OPENMM_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] OpenMM libraries not available: {e}")
    OPENMM_AVAILABLE = False

try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    print("[Warning] pdbfixer not installed.")
    PDBFIXER_AVAILABLE = False

class EnergyReporter:
    """Custom reporter to collect potential energies during simulation."""
    def __init__(self, interval=5000):
        self.interval = interval
        self.energies = []
    
    def describeNextReport(self, simulation):
        # Returns: (steps, position, velocity, force, energy, wrap)
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, True, False)

    def report(self, simulation, state):
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        self.energies.append(energy)

def run_short_md_evaluation(receptor_pdb, ligand_mol, output_prefix, reference_mol=None, sim_steps=2500000):
    """
    Runs MD with EXPLICIT SOLVENT (Water Box) + CUDA.
    Returns: avg_rmsd (A), std_rmsd (A), avg_energy (kcal/mol)
    """
    if not OPENMM_AVAILABLE:
        return None, None, None

    # Helper function to flush logs immediately to console
    def log(msg):
        print(msg)
        sys.stdout.flush()

    try:
        # Step 1: PDB Setup (Clean Receptor)
        if PDBFIXER_AVAILABLE:
            fixer = PDBFixer(filename=receptor_pdb)
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            topology, positions = fixer.topology, fixer.positions
        else:
            pdb = app.PDBFile(receptor_pdb)
            topology, positions = pdb.topology, pdb.positions

        # Step 2: Bond Order Repair (using Reference Ligand if available)
        if reference_mol:
            try:
                ref_heavy = Chem.RemoveHs(reference_mol)
                lig_heavy = Chem.RemoveHs(ligand_mol)
                fixed_rdkit_mol = AllChem.AssignBondOrdersFromTemplate(ref_heavy, lig_heavy)
                fixed_rdkit_mol = Chem.AddHs(fixed_rdkit_mol, addCoords=True)
            except:
                fixed_rdkit_mol = Chem.AddHs(ligand_mol, addCoords=True)
        else:
            fixed_rdkit_mol = Chem.AddHs(ligand_mol, addCoords=True)

        # Step 3-5: OpenFF & Topology Merging
        off_mol = Molecule.from_rdkit(fixed_rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        modeller = app.Modeller(topology, positions)
        modeller.add(off_mol.to_topology().to_openmm(), off_mol.conformers[0].to_openmm())
        modeller.addHydrogens()
        
        # Step 6: ForceField for EXPLICIT SOLVENT
        # Load Standard Amber14 (Protein) + TIP3P (Water)
        log(" [MD] Loading ForceField (Amber14 + TIP3P)...")
        try:
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        except Exception as e:
            log(f" [MD Error] XML Load Failed: {e}")
            return None, None, None

        # Step 7: GAFF Registration (for Ligand)
        gaff = GAFFTemplateGenerator(molecules=off_mol)
        forcefield.registerTemplateGenerator(gaff.generator)

        # --- [CRITICAL] Add Water Box ---
        log(" [MD] Adding Solvent Box (Padding=1.0nm)...")
        # Padding 1.0 nm ensures the protein is well solvated
        modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
        # --------------------------------

        # Step 8: Create System (PME for Explicit Solvent)
        # We do NOT pass implicitSolvent here, avoiding the 'argument never used' error
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,       # Standard for explicit solvent
            nonbondedCutoff=1.0*unit.nanometer,
            constraints=app.HBonds
        )
        
        # Step 9: Platform Selection - USE CUDA
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
        
        platform = None
        prop = {}
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            prop = {'Precision': 'mixed'} # Mixed precision is faster
            log(f" [MD] Platform: CUDA (GPU) | Method: Explicit Solvent (PME)")
        except:
            # Fallback to CPU if CUDA fails
            try:
                platform = mm.Platform.getPlatformByName('CPU')
                log(f" [MD] Platform: CPU (Optimized) | Method: Explicit Solvent (PME)")
            except:
                log(" [MD Critical] No usable platform found.")
                return None, None, None

        simulation = app.Simulation(modeller.topology, system, integrator, platform, prop)
        simulation.context.setPositions(modeller.positions)
        
        # Minimize Energy (Remove Clashes)
        log(f" [MD] Minimizing (Removal of clashes)...")
        simulation.minimizeEnergy(maxIterations=1000)

        # Save Solvated PDB (Crucial for Analysis)
        # This file contains protein + ligand + water
        pdb_filename = f"{output_prefix}.pdb"
        with open(pdb_filename, 'w') as f:
            app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
        
        # Run Simulation
        simulation.reporters.append(app.DCDReporter(f"{output_prefix}.dcd", 5000))
        energy_reporter = EnergyReporter(interval=5000)
        simulation.reporters.append(energy_reporter)
        
        ns_time = (sim_steps * 2) / 1000000
        log(f" [MD] Running {ns_time} ns simulation ({sim_steps} steps)...")
        
        simulation.step(sim_steps)
        
        # --- Analysis Section (Topology-Based) ---
        if energy_reporter.energies:
            avg_energy = np.mean(energy_reporter.energies)
        else:
            avg_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        # 1. Load Trajectory using the Saved PDB
        traj = md.load_dcd(f"{output_prefix}.dcd", top=pdb_filename)
        
        # 2. Filter Water & Ions (Select only Protein + Ligand)
        # This query excludes HOH (water), NA/CL (ions) based on the PDB you sent.
        selection_query = "(not water) and (not resname HOH) and (not resname NA) and (not resname CL) and (not element Na) and (not element Cl)"
        atom_indices = traj.topology.select(selection_query)
        
        # Create a "Dry" trajectory containing only Protein and Ligand
        traj_dry = traj.atom_slice(atom_indices)
        
        # 3. Robust Index-based Selection
        # Logic: In Modeller, we added Receptor first, then Ligand last.
        # So the LAST residue in traj_dry must be the ligand.
        n_res = traj_dry.topology.n_residues
        
        if n_res > 1:
            # Get the last residue (Ligand)
            ligand_res = traj_dry.topology.residue(n_res - 1)
            
            # Select Ligand Atoms by Index in the dry trajectory
            ligand_indices = traj_dry.topology.select(f"resid {ligand_res.index}")
            
            # Select Protein Atoms (All indices EXCEPT Ligand)
            all_indices = np.arange(traj_dry.n_atoms)
            protein_indices = np.setdiff1d(all_indices, ligand_indices)
            
            # Filter for Heavy Atoms Only (Cleaner RMSD, ignores hydrogens)
            heavy_atoms = traj_dry.topology.select("element != H")
            
            # Intersection: (Protein & Heavy) and (Ligand & Heavy)
            protein_indices = np.intersect1d(protein_indices, heavy_atoms)
            ligand_indices = np.intersect1d(ligand_indices, heavy_atoms)

            if len(protein_indices) > 0 and len(ligand_indices) > 0:
                # Superpose (Align) on Protein
                traj_super = traj_dry.superpose(traj_dry[0], atom_indices=protein_indices)
                
                # Calculate Ligand RMSD
                rmsd_nm = md.rmsd(traj_super, traj_dry[0], atom_indices=ligand_indices)
                rmsd_ang = rmsd_nm * 10 # Convert to Angstrom
                
                avg_rmsd = np.mean(rmsd_ang)
                std_rmsd = np.std(rmsd_ang)
            else:
                log(f" [MD Warning] Atom selection failed. Protein atoms: {len(protein_indices)}, Ligand atoms: {len(ligand_indices)}")
                avg_rmsd, std_rmsd = None, None
        else:
            log(" [MD Warning] Dry trajectory has too few residues.")
            avg_rmsd, std_rmsd = None, None

        # Format Output
        rmsd_str = f"{avg_rmsd:.2f}" if avg_rmsd is not None else "N/A"
        std_str = f"{std_rmsd:.2f}" if std_rmsd is not None else "N/A"
        log(f" [MD] Done. RMSD={rmsd_str}±{std_str} A | E={avg_energy:.1f} kcal")
        
        return avg_rmsd, std_rmsd, avg_energy

    except Exception as e:
        log(f" [MD Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None