"""
examples/complex_example.py
============================
Minimal working example: prepare and run a protein–ligand complex
simulation using an AutoDock Vina output PDB.

Before running
--------------
1. Place your docked complex PDB in the working directory, e.g.
   ``/data/simulations/hsp90_ATP/hsp90_ATP.pdb``
   The ligand residue name in the PDB should match ``param_ligand_name``
   (default ``"UNL"`` for AutoDock Vina outputs).
2. Populate ``config/gmx/complex/`` with your MDP templates.
3. Ensure ``gmx``, ``antechamber``, and ``acpype`` are on your PATH.
4. Install dependencies:  ``pip install -r requirements.txt``
"""

from pathlib import Path
from sim_prep.complex import CplxSimPrepper

WORK_DIR     = Path("/data/simulations/hsp90_ATP")
PROTEIN_NAME = "hsp90_ATP"   # stem of the complex PDB
LIGAND_CODE  = "ATP"
MD_NAME      = "md_production"

sim = CplxSimPrepper(
    protein_name=PROTEIN_NAME,
    ligand_code=LIGAND_CODE,
    param_ligand_name="UNL",        # residue name used by AutoDock
    remove_ligands=["UNL"],         # residues to strip from the protein PDB
    sim_len=100,
    bx_dim=1.0,
    bx_shp="dodecahedron",
    md_name=MD_NAME,
    pos_ion="NA",
    neg_ion="CL",
    work_dir=WORK_DIR,
)

sim.validate_config()
sim.assign_attributes()
sim.copy_config_files()
sim.update_config_files()

# Ligand parameterisation and complex assembly
sim.process_autodocked_complex()   # clean PDB, extract ligand, pdb2gmx
sim.param_with_amber()             # AmberTools + ACPYPE → GROMACS topology
sim.gro2itp()                      # extract ITP from ligand TOP
sim.generate_ligand_ndx()          # heavy-atom index for the ligand
sim.build_gmx_complex()            # merge protein.gro + ligand.gro
sim.build_complex_topology()       # patch topol.top with ligand parameters

# Solvation and equilibration
sim.solvate()
sim.minimise_system()
sim.make_prot_lig_ndx()            # Protein_ATP + Water_and_ions index
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()
