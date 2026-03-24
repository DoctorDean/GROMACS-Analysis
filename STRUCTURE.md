# Repository Structure

```
GROMACS-Analysis/
│
├── README.md                   # Project overview and full usage guide
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Development guide and PR checklist
├── LIGAND_PARAMETERISATION.md  # AmberTools + ACPYPE setup and usage guide
├── STRUCTURE.md                # This file — repo layout reference
├── LICENSE                     # MIT licence
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes trajectories, MDP outputs, etc.
│
│── gromacs_analysis.py         # Post-simulation analysis toolkit
│                               #   GromacsAnalysis  – single entry point for
│                               #   all trajectory analysis, DCCM, PCA, FEL,
│                               #   network analysis, and PDB colouring
│
├── sim_prep/                   # Simulation preparation package
│   ├── __init__.py             # Re-exports ApoSimPrepper, CplxSimPrepper,
│   │                           #   MixMDPrepper for clean top-level imports
│   ├── base.py                 # SimulationPrepper – abstract base class
│   │                           #   shared GROMACS wrappers: pdb2gmx,
│   │                           #   editconf, solvate, genion, mdrun, …
│   ├── apo.py                  # ApoSimPrepper – protein-only systems
│   ├── complex.py              # CplxSimPrepper – protein–ligand complexes
│   │                           #   includes AmberTools/ACPYPE parameterisation
│   │                           #   and topology assembly
│   └── mixmd.py                # MixMDPrepper – mixed-solvent simulations
│                               #   multi-probe parameterisation and
│                               #   insert-molecules workflow
│
├── md-configs/                 # MDP template files 
│   └── gmx/
│       ├── apo/                # ions.mdp  em.mdp  nvt.mdp  npt.mdp  md.mdp
│       ├── complex/            # ions_prot_lig.mdp  em_prot_lig.mdp  …
│       └── mixmd/              # ions_mix.mdp  em_mix.mdp  …
│
├── utils/                      # Internal Python utilities
│   └── amber_params.py         # AmberParameteriser class
│                               #   native Python pipeline wrapping:
│                               #   obabel → antechamber → parmchk2
│                               #   → tleap → ACPYPE
│                               #   called by CplxSimPrepper and MixMDPrepper
│                               #   (replaces the former param_with_amber.sh)
│
└── examples/                   # Minimal working examples (no simulation data)
    ├── apo_example.py
    ├── complex_example.py
    └── mixmd_example.py
```

---

## Design principles

### Two independent modules

`gromacs_analysis.py` and `sim_prep/` solve different problems and have
no import dependency on each other.  You can use either one independently:

- Run a simulation with `sim_prep`, then analyse the output with
  `GromacsAnalysis`.
- Point `GromacsAnalysis` at trajectories produced by any other means
  (existing data, other software, etc.).

### `sim_prep/` inheritance hierarchy

```
SimulationPrepper  (sim_prep/base.py)  ← abstract base class
    │
    ├── ApoSimPrepper   (sim_prep/apo.py)
    ├── CplxSimPrepper  (sim_prep/complex.py)
    └── MixMDPrepper    (sim_prep/mixmd.py)
```

The base class owns all shared GROMACS subprocess calls (`_run`),
MDP patching helpers (`_patch_nsteps`), and the standard pipeline steps
(`solvate`, `minimise_system`, `nvt_equilibration`, `npt_equilibration`,
`production_run`).  Subclasses override steps that differ by simulation
type (e.g. `solvate` in `CplxSimPrepper` starts from `complex.gro`
rather than `newbox.gro`) and implement the two abstract methods
`validate_config` and `update_config_files`.

### `config/` 

MDP template files are system- and force-field-specific and often
contain values tuned for a particular HPC environment or research
project.  They are excluded by `.gitignore`.  The `config/gmx/`
directory layout shown above is the expected convention; populate it
with your own templates and set `config_dir` in each subclass if you
use a different layout.

### Connecting preparation → analysis

After `production_run()` completes, hand off to `GromacsAnalysis`:

```python
from sim_prep import ApoSimPrepper
from gromacs_analysis import GromacsAnalysis

# --- Preparation ---
sim = ApoSimPrepper(
    protein_name="hsp90",
    sim_len=100,
    bx_dim=1.0,
    bx_shp="dodecahedron",
    md_name="md_production",
    work_dir="/data/simulations/hsp90_apo",
)
sim.validate_config()
sim.assign_attributes()
# … run full pipeline …
sim.production_run()

# --- Analysis (same working directory) ---
analysis = GromacsAnalysis(
    md_name="md_production",
    protein_name="HSP90",
    work_dir="/data/simulations/hsp90_apo",
)
analysis.nopbc_and_fit()
analysis.essential_dynamics()
```
