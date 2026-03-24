"""
examples/mixmd_example.py
==========================
Minimal working example: prepare and run a mixed-solvent (MixMD)
simulation with two probe molecules.

Before running
--------------
1. Place your protein PDB in the working directory, e.g.
   ``/data/simulations/hsp90_mixmd/hsp90.pdb``
2. Populate ``config/gmx/mixmd/`` with your MDP templates.
3. Ensure ``gmx``, ``antechamber``, and ``acpype`` are on your PATH.
4. Install dependencies:  ``pip install -r requirements.txt``

Notes on probe concentrations
------------------------------
The ``number`` field in each ligand dict specifies how many copies of
the molecule are inserted into the box.  A rough guide for a typical
60×60×60 Å dodecahedron:

    ~100 copies ≈ 100 mM
    ~50  copies ≈ 50  mM

Adjust based on your box dimensions and the target co-solvent
concentration from the literature.
"""

from pathlib import Path
from base import MixMDPrepper

WORK_DIR     = Path("/data/simulations/hsp90_mixmd")
PROTEIN_NAME = "hsp90"
MD_NAME      = "md_production"

sim = MixMDPrepper(
    protein_name=PROTEIN_NAME,
    sim_len=100,
    bx_dim=1.5,               # slightly larger box to accommodate probe molecules
    bx_shp="dodecahedron",
    md_name=MD_NAME,
    pos_ion="NA",
    neg_ion="CL",
    ligands=[
        {
            "code":   "ACT",
            "number": 50,
            "smiles": "CC(=O)[O-]",
        },
        {
            "code":   "MSE",
            "number": 20,
            "smiles": "CSCC[C@@H](N)C(=O)O",
        },
    ],
    work_dir=WORK_DIR,
)

sim.validate_config()
sim.assign_attributes()
sim.copy_config_files()
sim.update_config_files()

# Protein preparation
sim.clean_pdb_file()
sim.protein_pdb2gmx()
sim.set_new_box()

# Ligand parameterisation
sim.param_all_ligands()    # AmberTools + ACPYPE for each probe
sim.top2itp()              # extract ITP files from TOP files
sim.merge_atomtypes()      # deduplicate [ atomtypes ] across all ITPs

# System assembly
sim.build_mixmd()          # insert-molecules for each probe
sim.solvate()

# Equilibration and production
sim.minimise_system()
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()
