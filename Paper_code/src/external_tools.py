import os
import subprocess
import glob

def run_cmd(cmd, log_file=None, timeout=None):
    """Executes command with optional timeout."""
    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.run(cmd, shell=True, check=True, stdout=f, stderr=f, timeout=timeout)
        else:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"   [Warning] Command timed out ({timeout}s): {cmd}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {cmd}")
        if e.stderr:
            try:
                print(f"[STDERR] {e.stderr.decode()}")
            except:
                pass
        return False

def convert_to_pdbqt(input_file, output_file, flexible=False):
    """Converts PDB/SDF to PDBQT using OpenBabel."""
    flags = "-xh --partialcharge gasteiger"
    if not flexible: flags += " -xr"
    run_cmd(f"obabel {input_file} -O {output_file} {flags}")

def run_pdb4amber(input_pdb, output_pdb):
    """
    Normalizes PDB for Amber:
    - Adds missing atoms
    - Fixes terminal residues (e.g. OXT -> C-term)
    - Removes non-standard residues
    """
    # pdb4amber output often appends _sslink, so we control the output name explicitly if possible
    # or we rename it after.
    # Standard usage: pdb4amber -i input.pdb -o output.pdb --dry
    cmd = f"pdb4amber -i {input_pdb} -o {output_pdb} --noter --dry" 
    # --dry: don't add water
    # --noter: don't write TER (TLeap handles it by chain)
    
    # We use a temp log
    if not run_cmd(cmd):
        print("   [Warning] pdb4amber failed. Using original PDB.")
        # Fallback copy
        import shutil
        shutil.copy(input_pdb, output_pdb)
        return False
    return True

def parameterize_ligand(input_mol_sdf, working_dir, output_name="LIG"):
    """
    Generates GAFF parameters.
    Switched default to GAFF (v1) for better stability with TLeap.
    """
    wd = os.path.abspath(working_dir)
    mol2 = os.path.join(wd, f"{output_name}.mol2")
    frcmod = os.path.join(wd, f"{output_name}.frcmod")
    ac_log = os.path.join(wd, f"antechamber_{output_name}.log")
    
    # Changed '-at gaff2' to '-at gaff' for robustness
    cmd_bcc = f"antechamber -i {input_mol_sdf} -fi sdf -o {mol2} -fo mol2 -c bcc -at gaff -rn LIG -dr no"
    cmd_neutral = cmd_bcc + " -nc 0"
    cmd_gas = f"antechamber -i {input_mol_sdf} -fi sdf -o {mol2} -fo mol2 -c gas -at gaff -rn LIG -nc 0 -dr no"

    success = False
    if not os.path.exists(mol2):
        print(f"   [Amber] Parameterizing {output_name} (Timeout 300s)...")
        # Extended timeout to 300s (5 min) as requested
        if run_cmd(cmd_bcc, log_file=ac_log, timeout=300):
            success = True
        else:
            print(f"   [Amber] AM1-BCC failed. Retrying with forced neutral charge...")
            if run_cmd(cmd_neutral, log_file=ac_log, timeout=300):
                success = True
            else:
                print(f"   [Amber] Neutral failed. Retrying with Gasteiger charges (Robust mode)...")
                if run_cmd(cmd_gas, log_file=ac_log, timeout=60):
                    success = True
    else:
        success = True

    if not success:
        print(f"   [Error] Ligand parameterization failed completely.")
        return None, None

    if os.path.exists(mol2) and not os.path.exists(frcmod):
        run_cmd(f"parmchk2 -i {mol2} -f mol2 -o {frcmod}")
            
    return mol2, frcmod

def build_amber_system(receptor_pdb, mol2, frcmod, output_prefix, working_dir):
    """Builds Amber system using TLeap."""
    if not mol2 or not frcmod: return None, None
    wd = os.path.abspath(working_dir)
    prmtop = os.path.join(wd, f"{output_prefix}.prmtop")
    inpcrd = os.path.join(wd, f"{output_prefix}.inpcrd")
    tleap_in = os.path.join(wd, f"tleap_{output_prefix}.in")
    tleap_log = os.path.join(wd, f"leap_{output_prefix}.log")
    
    # Switched 'leaprc.gaff2' to 'leaprc.gaff' to match Antechamber
    script = f"""
    source leaprc.protein.ff14SB
    source leaprc.water.tip3p
    source leaprc.gaff
    loadamberparams frcmod.ionsjc_tip3p
    loadamberparams {os.path.abspath(frcmod)}
    LIG = loadmol2 {os.path.abspath(mol2)}
    PROT = loadpdb {os.path.abspath(receptor_pdb)}
    COMPLEX = combine {{PROT LIG}}
    charge COMPLEX
    saveamberparm COMPLEX {prmtop} {inpcrd}
    quit
    """
    with open(tleap_in, "w") as f: f.write(script)
    
    # Run TLeap
    if not run_cmd(f"tleap -f {tleap_in}", log_file=tleap_log):
        # Only print error if it really failed
        print(f"   [Error] TLeap failed for {output_prefix}. Check {os.path.basename(tleap_log)}")
        return None, None
        
    if not os.path.exists(prmtop):
        print(f"   [Error] TLeap did not generate prmtop.")
        return None, None
        
    return prmtop, inpcrd

def run_smina_docking(receptor_pdbqt, ligand_pdbqt, ligand_sdf_ref, output_file, 
                      center=None, size=(20, 20, 20), 
                      autobox_ligand=None, exhaustiveness=8, custom_args=""):
    """Smart Smina Runner."""
    cmd = f"smina -r {receptor_pdbqt} -l {ligand_pdbqt} -o {output_file}"
    if autobox_ligand:
        if "--autobox_add" not in custom_args:
            cmd += f" --autobox_ligand {autobox_ligand} --autobox_add 8"
        else:
            cmd += f" --autobox_ligand {autobox_ligand}"
    elif center:
        cmd += f" --center_x {center[0]} --center_y {center[1]} --center_z {center[2]}"
        cmd += f" --size_x {size[0]} --size_y {size[1]} --size_z {size[2]}"
    
    if "--exhaustiveness" not in custom_args and "--minimize" not in custom_args:
        cmd += f" --exhaustiveness {exhaustiveness}"
    if custom_args: 
        cmd += f" {custom_args}"
        
    run_cmd(cmd)