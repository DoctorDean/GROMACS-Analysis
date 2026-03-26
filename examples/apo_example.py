"""
examples/apo_example.py
=======================
Minimal working example: prepare and run an apo protein simulation,
then hand off to GromacsAnalysis for post-simulation analysis.

Before running
--------------
1. Place your protein PDB in the working directory, e.g.
   ``/data/simulations/hsp90_apo/hsp90.pdb``
2. Populate ``configs/apo/`` with your MDP templates.
3. Ensure ``gmx`` is on your PATH.
4. Install dependencies:  ``pip install -r requirements.txt``
"""

from pathlib import Path
from sim_prep.apo import ApoSimPrepper
from gromacs_analysis import GromacsAnalysis

WORK_DIR     = Path("/data/simulations/hsp90_apo")
PROTEIN_NAME = "hsp90"
MD_NAME      = "md_production"

# ── 1. Simulation preparation ────────────────────────────────────────────────

sim = ApoSimPrepper(
    protein_name=PROTEIN_NAME,
    sim_len=100,              # nanoseconds
    bx_dim=1.0,               # nm padding
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

sim.clean_pdb_file()
sim.protein_pdb2gmx()
sim.set_new_box()
sim.solvate()
sim.minimise_system()
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()

# ── 2. Post-simulation analysis ───────────────────────────────────────────────

analysis = GromacsAnalysis(
    md_name=MD_NAME,
    protein_name="HSP90",
    work_dir=WORK_DIR,
)

analysis.nopbc_and_fit()
analysis.essential_dynamics(time_unit="ns")
analysis.covariance_analysis(time_unit="ns", last=10)

corr = analysis.covariance_to_correlation()
analysis.plot_dccm(corr=corr, cmap="bwr")

G = analysis.correlation_network(corr=corr, threshold=0.3)
analysis.plot_correlation_network(G)

pc_data   = analysis.project_pca(first=1, last=2)
landscape = analysis.free_energy_landscape(pc_data=pc_data, temperature=300.0)
analysis.plot_free_energy_3d(landscape)

analysis.colour_pdb_by_rmsf()
for metric in ["betweenness", "closeness"]:
    analysis.colour_pdb_by_centrality(graph=G, metric=metric)
