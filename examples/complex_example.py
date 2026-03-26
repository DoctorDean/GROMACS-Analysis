"""
examples/complex_example.py
============================
Working examples for all three CplxSimPrepper input paths.

CplxSimPrepper supports three ways of providing your starting structures.
Choose the path that matches your upstream workflow — all three converge
at param_with_amber() and share identical solvation, equilibration, and
production steps from that point on.

    Path A — AutoDock complex PDB
        You have a single PDB containing both protein and docked ligand,
        as produced by AutoDock Vina, Smina, or similar tools.
        The ligand residue is typically named "UNL" or "UNK".

    Path B — Native structure + separate ligand
        You have a protein structure from the RCSB PDB, AlphaFold, or
        similar source (.pdb, .cif, .mmcif) and a ligand in a separate
        file (.sdf, .mol2, .pdb) or as a SMILES string.
        This path handles format conversion and structure cleaning
        automatically.

    Path C — Pre-separated files
        You have already separated and cleaned your protein and ligand
        into individual files (e.g. from a Schrödinger Glide workflow
        or manual preparation). This is a thin wrapper around Path B
        that makes intent explicit in your pipeline scripts.

Before running any path
-----------------------
1. Populate md-configs/complex/ with your MDP templates.
2. Ensure gmx, antechamber, parmchk2, tleap, acpype, and obabel are
   on your PATH. See LIGAND_PARAMETERISATION.md for setup instructions.
3. Install dependencies:  pip install -r requirements.txt
"""

from pathlib import Path
from sim_prep import CplxSimPrepper

# =============================================================================
# Shared simulation parameters — used by all three paths
# =============================================================================

SHARED_PARAMS = dict(
    sim_len  = 100,             # nanoseconds
    bx_dim   = 1.0,             # nm box padding
    bx_shp   = "dodecahedron",
    md_name  = "md_production",
    pos_ion  = "NA",
    neg_ion  = "CL",
)

# Shared tail: solvation → equilibration → production
# Called identically at the end of every path — defined once here so the
# examples below stay concise and don't repeat themselves.
def run_shared_pipeline(sim: CplxSimPrepper) -> None:
    sim.solvate()
    sim.minimise_system()
    sim.make_prot_lig_ndx()     # builds Protein_<ligand_code> + Water_and_ions index
    sim.nvt_equilibration()
    sim.npt_equilibration()
    sim.production_run()


# =============================================================================
# PATH A — AutoDock / Smina docked complex
# =============================================================================
#
# Input:  a single PDB containing both protein and docked ligand,
#         with the ligand residue named "UNL" (AutoDock Vina default)
#         or "UNK" depending on the docking software.
#
# File layout expected in WORK_DIR:
#   hsp90_ATP.pdb    ← docked complex from AutoDock Vina
#
# process_autodocked_complex() will:
#   - strip "UNL" residues → hsp90_ATP_clean.pdb  (protein for pdb2gmx)
#   - extract and rename to → ATP.pdb             (ligand for parameterisation)
#   - run gmx pdb2gmx on the cleaned protein

def path_a_autodock():
    WORK_DIR = Path("/data/simulations/hsp90_ATP/autodock")

    sim = CplxSimPrepper(
        protein_name    = "hsp90_ATP",   # must match the complex PDB stem
        ligand_code     = "ATP",
        param_ligand_name = "UNL",       # residue name in the docked PDB
        remove_ligands  = ["UNL"],       # residues stripped from the protein PDB
        work_dir        = WORK_DIR,
        **SHARED_PARAMS,
    )

    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()

    # ── Path A specific ───────────────────────────────────────────────
    sim.process_autodocked_complex()    # split, clean, pdb2gmx

    # ── Converged pipeline (same for all paths) ───────────────────────
    sim.param_with_amber()              # obabel → antechamber → tleap → ACPYPE
    sim.gro2itp()                       # extract ITP from ligand TOP
    sim.generate_ligand_ndx()           # heavy-atom index for the ligand
    sim.build_gmx_complex()             # merge protein.gro + ligand.gro
    sim.build_complex_topology()        # patch topol.top with ligand params
    run_shared_pipeline(sim)


# =============================================================================
# PATH B — Native structure from RCSB / AlphaFold + separate ligand
# =============================================================================
#
# Suitable when:
#   - You downloaded a structure from the RCSB PDB (modern .cif format)
#   - You have an AlphaFold model (.cif) with a ligand from a database
#   - Your ligand comes from PubChem / ChEMBL as an SDF, or as a SMILES
#
# prepare_from_structure() handles:
#   - mmCIF → PDB conversion (via gemmi)
#   - Stripping waters, HETATM records, alternate locations, ANISOU
#   - Retaining specified metal ions or co-factors (keep_residues)
#   - Ligand conversion from SDF / MOL2 / SMILES → PDB (via RDKit)
#
# Three sub-variants are shown below.

def path_b_rcsb_cif_with_sdf():
    """
    RCSB mmCIF structure + SDF ligand from PubChem or ChEMBL.
    Typical for structure-based drug design starting from a known crystal.
    """
    WORK_DIR = Path("/data/simulations/6lu7_N3/rcsb")

    sim = CplxSimPrepper(
        protein_name = "6lu7",
        ligand_code  = "N3I",
        work_dir     = WORK_DIR,
        **SHARED_PARAMS,
    )

    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()

    # ── Path B specific ───────────────────────────────────────────────
    sim.prepare_from_structure(
        protein_input  = "6lu7.cif",         # RCSB mmCIF download
        ligand_input   = "N3_inhibitor.sdf", # SDF from PubChem
        keep_residues  = ["ZN"],             # retain the catalytic zinc ion
        remove_waters  = True,
        remove_hetatm  = True,
    )

    # ── Converged pipeline ────────────────────────────────────────────
    sim.param_with_amber(net_charge=-1)     # N3 carries a net -1 charge
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()
    run_shared_pipeline(sim)


def path_b_alphafold_with_smiles():
    """
    AlphaFold model (.cif) with a ligand provided as a SMILES string.
    No ligand file needed — RDKit generates 3-D coordinates with ETKDG
    and minimises with MMFF94.
    Useful for virtual screening hits where only a SMILES is available.
    """
    WORK_DIR = Path("/data/simulations/p53_PAR/alphafold")

    sim = CplxSimPrepper(
        protein_name = "p53_AF",
        ligand_code  = "PAR",
        work_dir     = WORK_DIR,
        **SHARED_PARAMS,
    )

    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()

    # ── Path B specific ───────────────────────────────────────────────
    sim.prepare_from_structure(
        protein_input = "AF-P04637-F1-model_v4.cif",   # AlphaFold download
        ligand_input  = "smiles:CC(=O)Nc1ccc(O)cc1",   # paracetamol SMILES
    )

    # ── Converged pipeline ────────────────────────────────────────────
    sim.param_with_amber()
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()
    run_shared_pipeline(sim)


def path_b_plain_pdb_with_mol2():
    """
    Standard PDB structure + MOL2 ligand.
    Typical when starting from a homology model or a PDB file that has
    already been cleaned outside this pipeline, with a ligand from
    Glide or GOLD in MOL2 format.
    """
    WORK_DIR = Path("/data/simulations/hsp90_ADP/pdb_mol2")

    sim = CplxSimPrepper(
        protein_name = "hsp90",
        ligand_code  = "ADP",
        work_dir     = WORK_DIR,
        **SHARED_PARAMS,
    )

    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()

    # ── Path B specific ───────────────────────────────────────────────
    sim.prepare_from_structure(
        protein_input = "hsp90_homology.pdb",
        ligand_input  = "ADP_docked.mol2",    # Glide or GOLD MOL2 output
    )

    # ── Converged pipeline ────────────────────────────────────────────
    sim.param_with_amber(net_charge=-3)       # ADP carries a net -3 charge
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()
    run_shared_pipeline(sim)


# =============================================================================
# PATH C — Pre-separated protein and ligand files
# =============================================================================
#
# Suitable when:
#   - Your upstream workflow (e.g. Schrödinger Maestro, manual prep) already
#     delivers protein and ligand as separate, clean files
#   - You want to be explicit in your script that the files are already split
#
# prepare_from_separate_files() accepts exactly the same file formats as
# Path B (protein: .pdb/.cif/.mmcif/.gro, ligand: .pdb/.mol2/.sdf/.mol
# or "smiles:...") and applies identical conversion and cleaning.
#
# File layout expected in WORK_DIR:
#   hsp90_prepared.pdb   ← protein-only, pre-cleaned (e.g. from Maestro)
#   ATP_glide.sdf        ← ligand-only, from Glide SP docking output

def path_c_preseparated():
    WORK_DIR = Path("/data/simulations/hsp90_ATP/glide")

    sim = CplxSimPrepper(
        protein_name = "hsp90",
        ligand_code  = "ATP",
        work_dir     = WORK_DIR,
        **SHARED_PARAMS,
    )

    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()

    # ── Path C specific ───────────────────────────────────────────────
    sim.prepare_from_separate_files(
        protein_file  = "hsp90_prepared.pdb",
        ligand_file   = "ATP_glide.sdf",
        keep_residues = ["MG"],    # retain magnesium co-factor
    )

    # ── Converged pipeline ────────────────────────────────────────────
    sim.param_with_amber(net_charge=-4)    # ATP carries a net -4 charge
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()
    run_shared_pipeline(sim)


# =============================================================================
# Run a specific path
# =============================================================================
# Uncomment the path that matches your input data, or call the function
# directly from another script:
#
#   from examples.complex_example import path_b_rcsb_cif_with_sdf
#   path_b_rcsb_cif_with_sdf()

if __name__ == "__main__":
    # path_a_autodock()
    # path_b_rcsb_cif_with_sdf()
    # path_b_alphafold_with_smiles()
    # path_b_plain_pdb_with_mol2()
    # path_c_preseparated()
    pass