# MDAnalysis-GROMACS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-%E2%89%A52019-green.svg)](https://www.gromacs.org/)

An integrated Python toolkit for GROMACS molecular dynamics simulation preparation and trajectory analysis. Covers the full pipeline — from raw protein structure to publication-quality figures — in a single, reproducible Python script.

The package has two independent modules that can be used together or separately:

- **`sim_prep`** — automates GROMACS simulation setup for apo proteins, protein–ligand complexes, and mixed-solvent (MixMD) simulations
- **`GromacsAnalysis`** — wraps 16 GROMACS analysis commands and extends them with pure-Python DCCM, PCA, free energy landscape, graph-theoretic network analysis, and PDB structure annotation

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simulation Preparation](#simulation-preparation)
  - [Configuration Files](#configuration-files)
  - [Apo Simulations](#apo-simulations)
  - [Protein–Ligand Complex](#proteinligand-complex)
  - [Mixed-Solvent MixMD](#mixed-solvent-mixmd)
  - [HPC Checkpointing](#hpc-checkpointing)
- [Trajectory Analysis](#trajectory-analysis)
  - [Essential Dynamics](#essential-dynamics)
  - [Hydrogen Bond Analysis](#hydrogen-bond-analysis)
  - [SASA Analysis](#sasa-analysis)
  - [Covariance and DCCM](#covariance-and-dccm)
  - [Correlation Network](#correlation-network)
  - [Free Energy Landscape](#free-energy-landscape)
  - [PDB Colouring](#pdb-colouring)
  - [Multi-System Comparison](#multi-system-comparison)
- [API Reference](#api-reference)
- [Output Files](#output-files)
- [Visualising Results](#visualising-results)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

### Simulation preparation (`sim_prep`)

| Class | Simulation type | Key steps |
|---|---|---|
| `ApoSimPrepper` | Protein-only | PDB cleaning → pdb2gmx → solvation → equilibration → production |
| `CplxSimPrepper` | Protein–ligand complex | 3 input paths → GAFF parameterisation → topology assembly → equilibration |
| `MixMDPrepper` | Mixed-solvent (MixMD) | Multi-probe parameterisation → insert-molecules → equilibration |

- YAML / JSON config file support — drive the full pipeline from a plain text file
- Persistent HPC checkpointing — resume interrupted pipelines from the last completed step
- Native Python ligand parameterisation (Open Babel → antechamber → parmchk2 → tleap → ACPYPE)
- Structure format conversion: `.pdb`, `.cif` / `.mmcif`, `.gro`, `.mol2`, `.sdf`, SMILES strings

### Trajectory analysis (`GromacsAnalysis`)

| Category | Method | Description |
|---|---|---|
| Trajectory prep | `nopbc_and_fit()` | Remove PBC, least-squares rot+trans fit |
| Essential dynamics | `essential_dynamics()` | RMSD, RMSF, radius of gyration |
| H-bond analysis | `hbond_analysis()` | H-bond count over time + per-pair occupancy |
| H-bond plot | `plot_hbond_occupancy()` | Occupancy bar chart + time series |
| SASA | `sasa_analysis()` | Total and per-residue solvent accessible surface area |
| SASA plot | `plot_sasa()` | SASA time series + per-residue colour-mapped bar chart |
| Covariance / PCA | `covariance_analysis()` | Full `gmx covar` wrapper |
| DCCM | `covariance_to_correlation()` | Covariance → normalised cross-correlation matrix |
| DCCM plot | `plot_dccm()` | Heatmap with configurable colourmap and tick spacing |
| Network | `correlation_network()` | Weighted residue graph + 10 graph-theoretic metrics |
| Network plot | `plot_correlation_network()` | Degree-centrality-coloured spring layout |
| PC projections | `project_pca()` | `gmx anaeig` → tidy CSV |
| Free energy | `free_energy_landscape()` | 2-D FEL via Boltzmann inversion + Gaussian smoothing |
| FEL plot | `plot_free_energy_3d()` | 3-D surface + gradient magnitude heatmap with contours |
| PDB colour | `colour_pdb_by_rmsf()` | B-factor ← RMSF |
| PDB colour | `colour_pdb_by_centrality()` | B-factor ← degree / betweenness / closeness / eigenvector |

---

## Repository Structure

```
mdanalysis-gromacs/
│
├── README.md                        # This file
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Development guide and PR checklist
├── LICENSE                          # MIT licence
├── LIGAND_PARAMETERISATION.md       # AmberTools + ACPYPE setup guide
├── STRUCTURE.md                     # Detailed repo layout and design notes
├── pyproject.toml                   # PEP 517/518 build config; pip install .
├── requirements.txt                 # Tiered dependency list
│
├── docs/
│   ├── paper.md                     # JOSS manuscript
│   └── paper.bib                    # BibTeX references
│
├── gromacs_analysis.py              # Post-simulation analysis
│   └── GromacsAnalysis              # Single class — all analysis methods
│       ├── nopbc_and_fit()
│       ├── essential_dynamics()
│       ├── hbond_analysis() / plot_hbond_occupancy()
│       ├── sasa_analysis() / plot_sasa()
│       ├── covariance_analysis()
│       ├── covariance_to_correlation() / plot_dccm()
│       ├── correlation_network() / plot_correlation_network()
│       ├── project_pca()
│       ├── free_energy_landscape() / plot_free_energy_3d()
│       └── colour_pdb_by_rmsf() / colour_pdb_by_centrality()
│
├── sim_prep/                        # Simulation preparation package
│   ├── __init__.py                  # Exports ApoSimPrepper, CplxSimPrepper, MixMDPrepper
│   ├── base.py                      # SimulationPrepper — abstract base class
│   │                                #   shared steps: pdb2gmx, editconf, solvate,
│   │                                #   genion, minimise, nvt, npt, production
│   ├── apo.py                       # ApoSimPrepper — protein-only simulations
│   ├── complex.py                   # CplxSimPrepper — protein–ligand complexes
│   │                                #   Path A: AutoDock complex PDB
│   │                                #   Path B: native PDB/mmCIF + ligand file/SMILES
│   │                                #   Path C: pre-separated protein + ligand files
│   ├── mixmd.py                     # MixMDPrepper — mixed-solvent simulations
│   ├── config.py                    # YAML/JSON config loader + template generator
│   └── checkpoint.py               # CheckpointMixin — HPC resume support
│
├── utils/                           # Internal utilities
│   ├── __init__.py
│   ├── amber_params.py              # AmberParameteriser — ligand parameterisation
│   │                                #   obabel → antechamber → parmchk2 → tleap → ACPYPE
│   └── structure_io.py              # prepare_structure() — format conversion
│                                    #   .pdb .cif .mmcif .gro .mol2 .sdf smiles:
│
├── config/
│   ├── gmx/                         # MDP template files (not committed — user-provided)
│   │   ├── apo/                     # ions.mdp  em.mdp  nvt.mdp  npt.mdp  md.mdp
│   │   ├── complex/                 # ions_prot_lig.mdp  em_prot_lig.mdp  ...
│   │   └── mixmd/                   # ions_mix.mdp  em_mix.mdp  ...
│   └── examples/                    # Annotated YAML reference configs
│       ├── apo.yaml
│       ├── complex.yaml
│       └── mixmd.yaml
│
├── examples/                        # Working example scripts
│   ├── apo_example.py               # Full apo pipeline
│   ├── complex_example.py           # All three complex input paths
│   └── mixmd_example.py             # MixMD with multiple probes
│
└── tests/
    └── test_suite.py                # 53 tests across 14 classes
```

> **Note on `config/`:** MDP template files contain force-field and
> hardware-specific parameters but those committed to the repository may not be
> set up for your system. Populate this directory with your own templates. 
> See `config/examples/` for the expected file names and `CONTRIBUTING.md` 
> for the recommended MDP parameter values.

---

## Requirements

### External software

| Tool | Version | Required for |
|---|---|---|
| [GROMACS](https://www.gromacs.org/) | ≥ 2019 | All simulation and analysis steps |
| [AmberTools](https://ambermd.org/AmberTools.php) | ≥ 23 | Ligand parameterisation (`antechamber`, `parmchk2`, `tleap`) |
| [ACPYPE](https://github.com/alanwilter/acpype) | ≥ 2022.7 | Amber → GROMACS topology conversion |
| [Open Babel](https://openbabel.org/) | ≥ 3.1 | Ligand protonation |

AmberTools, ACPYPE, and Open Babel are only required for `CplxSimPrepper` and `MixMDPrepper`. See [LIGAND_PARAMETERISATION.md](LIGAND_PARAMETERISATION.md) for full installation instructions.

### Python

- Python ≥ 3.10

**Core dependencies** (always required):

```
biopython  matplotlib  networkx  numpy  pandas  pyyaml  scipy
```

**Structure handling** (required for non-PDB inputs):

```
gemmi  rdkit
```

Best installed via conda: `conda install -c conda-forge rdkit gemmi`

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/DoctorDean/GROMACS-Analysis.git
cd mdanalysis-gromacs
pip install -e .
```

The `-e` flag installs in editable mode so changes to the source are reflected immediately without reinstalling.

### Core dependencies only

```bash
pip install -e .
```

### With structure handling support

```bash
pip install -e ".[structure]"
# Or via conda for RDKit:
conda install -c conda-forge rdkit gemmi
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

> **GROMACS is not installed via pip.** Install it via your system package
> manager, HPC module system, or follow the
> [official installation guide](https://manual.gromacs.org/documentation/current/install-guide/index.html).

---

## Quick Start

### Analysis only (existing trajectory)

```python
from gromacs_analysis import GromacsAnalysis

analysis = GromacsAnalysis(
    md_name="md_production",    # stem of your .tpr / .xtc files
    protein_name="MyProtein",   # used to label all output files
    work_dir="/path/to/sim",    # directory containing simulation files
)

analysis.nopbc_and_fit()
analysis.essential_dynamics(time_unit="ns")
analysis.hbond_analysis()
analysis.sasa_analysis()
```

### Preparation from a YAML config file

```bash
# Generate an annotated template
python -c "from sim_prep.config import generate_template; generate_template('apo', 'my_sim.yaml')"

# Edit my_sim.yaml, then run:
python -c "from sim_prep.config import load_config; load_config('my_sim.yaml', run_all=True)"
```

---

## Simulation Preparation

### Configuration Files

Every simulation type can be fully configured from a YAML or JSON file.
Annotated templates are provided in `config/examples/`.

```python
from sim_prep.config import load_config, generate_template

# Write an annotated template to disk
generate_template("apo",     "apo_sim.yaml")
generate_template("complex", "complex_sim.yaml")
generate_template("mixmd",   "mixmd_sim.yaml")

# Load and run
sim = load_config("apo_sim.yaml")              # returns validated prepper object
sim = load_config("apo_sim.yaml", run_all=True)  # runs full pipeline immediately
```

A minimal apo config:

```yaml
type: apo
protein_name: hsp90
sim_len: 100
bx_shp: dodecahedron
bx_dim: 1.0
md_name: md_production
pos_ion: NA
neg_ion: CL
work_dir: /data/simulations/hsp90_apo
```

### Apo Simulations

```python
from sim_prep import ApoSimPrepper

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
```

### Protein–Ligand Complex

Three input paths are supported — choose the one that matches your upstream workflow:

```python
from sim_prep import CplxSimPrepper

sim = CplxSimPrepper(
    protein_name="hsp90_ATP",
    ligand_code="ATP",
    sim_len=100,
    bx_dim=1.0,
    bx_shp="dodecahedron",
    md_name="md_production",
    work_dir="/data/simulations/hsp90_ATP",
)
sim.validate_config()
sim.assign_attributes()
sim.copy_config_files()
sim.update_config_files()
```

**Path A — AutoDock / Smina docked complex:**

```python
sim.process_autodocked_complex()   # split protein + ligand, run pdb2gmx
```

**Path B — Native PDB / mmCIF + separate ligand:**

```python
sim.prepare_from_structure(
    protein_input="6lu7.cif",        # .pdb / .cif / .mmcif / .gro
    ligand_input="N3.sdf",           # .pdb / .mol2 / .sdf / smiles:CC...
    keep_residues=["ZN"],            # retain metal ions
)
```

**Path C — Pre-separated files (Glide, manual prep):**

```python
sim.prepare_from_separate_files(
    protein_file="protein_prepared.pdb",
    ligand_file="ligand_glide.sdf",
)
```

**All paths continue identically from here:**

```python
sim.param_with_amber(net_charge=-4)  # net_charge is critical for charged ligands
sim.gro2itp()
sim.generate_ligand_ndx()
sim.build_gmx_complex()
sim.build_complex_topology()
sim.solvate()
sim.minimise_system()
sim.make_prot_lig_ndx()
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()
```

| Upstream tool | Path | Input |
|---|---|---|
| AutoDock Vina / Smina | A | Docked complex PDB |
| RCSB PDB / AlphaFold | B | `.cif` + ligand `.sdf` or SMILES |
| Schrödinger Glide | C | Separate protein + ligand SDF |
| Virtual screening hit | B | Protein `.pdb` + SMILES string |

### Mixed-Solvent MixMD

```python
from sim_prep import MixMDPrepper

sim = MixMDPrepper(
    protein_name="hsp90",
    sim_len=100,
    bx_dim=1.5,
    bx_shp="dodecahedron",
    md_name="md_production",
    work_dir="/data/simulations/hsp90_mixmd",
    ligands=[
        {"code": "ACT", "number": 50, "smiles": "CC(=O)[O-]"},
        {"code": "BNZ", "number": 20, "smiles": "c1ccccc1"},
    ],
)
sim.validate_config()
sim.assign_attributes()
sim.copy_config_files()
sim.update_config_files()
sim.clean_pdb_file()
sim.protein_pdb2gmx()
sim.set_new_box()
sim.param_all_ligands(net_charges={"ACT": -1, "BNZ": 0})
sim.top2itp()
sim.merge_atomtypes()
sim.build_mixmd()
sim.solvate()
sim.minimise_system()
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()
```

### HPC Checkpointing

On HPC clusters where jobs have wall-time limits, the `CheckpointMixin` allows a
pipeline to resume from the last completed step rather than restarting from scratch.

```python
from sim_prep.checkpoint import CheckpointMixin
from sim_prep import ApoSimPrepper

class CheckpointedApo(CheckpointMixin, ApoSimPrepper):
    pass

sim = CheckpointedApo(
    protein_name="hsp90",
    sim_len=100,
    bx_dim=1.0,
    bx_shp="dodecahedron",
    md_name="md_production",
    work_dir="/data/simulations/hsp90_apo",
)
sim.validate_config()
sim.assign_attributes()
sim.resume_from_checkpoint()   # loads prior state; completed steps are skipped

sim.clean_pdb_file()           # skipped if already done
sim.protein_pdb2gmx()          # skipped if already done
sim.set_new_box()              # resumes from here if this was the failure point
sim.solvate()
sim.minimise_system()
sim.nvt_equilibration()
sim.npt_equilibration()
sim.production_run()
```

State is written to `.sim_checkpoint.json` in `work_dir` after every completed step. To check status or reset:

```python
sim.checkpoint_status()            # print completed / pending steps
sim.reset_checkpoint(["solvate"])  # re-run a specific step
sim.reset_checkpoint()             # full reset
```

---

## Trajectory Analysis

### Essential Dynamics

```python
from gromacs_analysis import GromacsAnalysis, SelectionGroups

groups = SelectionGroups(
    center="1",         # Protein — centering group
    output="0",         # System  — output group
    fit="4",            # Backbone — for rot+trans fitting
    rmsf="4",           # Backbone — RMSF per residue
    rg="1",             # Protein  — radius of gyration
    covar_fit="4",
    covar_analysis="4",
)

analysis = GromacsAnalysis(
    md_name="md_production",
    protein_name="HSP90",
    work_dir="/data/simulations/hsp90_apo",
    groups=groups,
)

analysis.nopbc_and_fit()
analysis.essential_dynamics(time_unit="ns")
```

### Hydrogen Bond Analysis

```python
# Intra-protein H-bonds
analysis.hbond_analysis(donor_group="Protein", acceptor_group="Protein")

# Protein–ligand interface H-bonds
analysis.hbond_analysis(donor_group="Protein", acceptor_group="ATP")

# Plot occupancy bar chart + time series
analysis.plot_hbond_occupancy(top_n=20, occupancy_threshold=0.1)
```

Outputs: `hbond_num_{name}.xvg`, `hbond_matrix_{name}.xpm`,
`hbond_occupancy_{name}.csv`, `hbond_occupancy_{name}.png`,
`hbond_timeseries_{name}.png`

### SASA Analysis

```python
analysis.sasa_analysis(group="Protein", probe_radius=0.14)
analysis.plot_sasa()
```

Outputs: `sasa_total_{name}.xvg`, `sasa_residue_{name}.csv`,
`sasa_timeseries_{name}.png`, `sasa_residue_{name}.png`

### Covariance and DCCM

```python
analysis.covariance_analysis(time_unit="ns", last=10)
corr = analysis.covariance_to_correlation()
analysis.plot_dccm(corr=corr, cmap="bwr")
```

### Correlation Network

```python
G = analysis.correlation_network(corr=corr, threshold=0.3)
analysis.plot_correlation_network(G)
```

Printed metrics: connectivity, average path length, average clustering,
max degree / betweenness / closeness / eigenvector centrality, max edge
betweenness, degree assortativity, k-core size, articulation points.

### Free Energy Landscape

```python
pc_data   = analysis.project_pca(first=1, last=2, time_unit="ns")
landscape = analysis.free_energy_landscape(
    pc_data=pc_data,
    bin_width=2.0,
    sigma=1.5,
    temperature=300.0,
)
analysis.plot_free_energy_3d(landscape, cmap_G="viridis", cmap_grad="plasma")
```

### PDB Colouring

```python
# Colour by per-residue flexibility
analysis.colour_pdb_by_rmsf(pdb_file="average_structure.pdb")

# Colour by network centrality
for metric in ["degree", "betweenness", "closeness", "eigenvector"]:
    analysis.colour_pdb_by_centrality(
        pdb_file="average_structure.pdb",
        graph=G,
        metric=metric,
    )
```

Open in PyMOL: `spectrum b, blue_white_red`
Open in ChimeraX: `color bfactor palette blue:white:red`

### Multi-System Comparison

Each `GromacsAnalysis` instance is self-contained, making side-by-side
comparisons straightforward:

```python
systems = {
    "apo":  GromacsAnalysis("md_production", "HSP90_apo",  work_dir="/data/apo"),
    "holo": GromacsAnalysis("md_production", "HSP90_holo", work_dir="/data/holo"),
}

for name, sim in systems.items():
    sim.nopbc_and_fit()
    sim.essential_dynamics()
    sim.hbond_analysis()
    sim.sasa_analysis()
    corr = sim.covariance_to_correlation()
    G    = sim.correlation_network(corr=corr)
    pc   = sim.project_pca()
    land = sim.free_energy_landscape(pc_data=pc, temperature=300.0)
    sim.plot_free_energy_3d(land)
    sim.colour_pdb_by_centrality(graph=G, metric="betweenness")
```

---

## API Reference

### `SelectionGroups`

Dataclass holding GROMACS selection group indices for each analysis step.
All fields default to standard protein-in-water indices.

| Field | Default | Group | Used by |
|---|---|---|---|
| `center` | `"1"` | Protein | `nopbc_and_fit` — centring |
| `output` | `"0"` | System | `nopbc_and_fit` — output |
| `fit` | `"1"` | Protein | `nopbc_and_fit`, `essential_dynamics` |
| `rmsf` | `"4"` | Backbone | `essential_dynamics` |
| `rg` | `"1"` | Protein | `essential_dynamics` |
| `covar_fit` | `"4"` | Backbone | `covariance_analysis` |
| `covar_analysis` | `"4"` | Backbone | `covariance_analysis` |

### `GromacsAnalysis`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `md_name` | `str` | — | Stem of `.tpr` / `.xtc` files |
| `protein_name` | `str` | — | Label used in all output filenames |
| `work_dir` | `str \| Path` | `cwd` | Directory containing simulation files |
| `groups` | `SelectionGroups` | defaults | GROMACS selection group indices |
| `index_file` | `str` | `None` | Path to custom `.ndx` file |
| `gmx_executable` | `str` | `"gmx"` | GROMACS binary name or full path |

Full method signatures and parameter documentation are available via:

```python
help(GromacsAnalysis.hbond_analysis)
help(GromacsAnalysis.free_energy_landscape)
# etc.
```

### `sim_prep` classes

| Class | Required kwargs | Key optional kwargs |
|---|---|---|
| `ApoSimPrepper` | `protein_name`, `sim_len`, `bx_dim`, `bx_shp`, `md_name` | `pos_ion`, `neg_ion`, `work_dir`, `gmx_executable`, `index_file` |
| `CplxSimPrepper` | + `ligand_code` | + `param_ligand_name`, `remove_ligands`, `net_charge`, `atom_type` |
| `MixMDPrepper` | + `ligands` (list of dicts) | + `net_charges` (dict) |

---

## Output Files

### Analysis outputs

| File | Method | Description |
|---|---|---|
| `{md}_noPBC.xtc` | `nopbc_and_fit` | PBC-corrected trajectory |
| `{md}_fitted.xtc` | `nopbc_and_fit` | Fitted trajectory |
| `rmsd_{name}.xvg` | `essential_dynamics` | Backbone RMSD vs time |
| `rmsf_{name}.xvg` | `essential_dynamics` | Per-residue RMSF |
| `rg_{name}.xvg` | `essential_dynamics` | Radius of gyration vs time |
| `average_structure_{name}.pdb` | `essential_dynamics` | Time-averaged structure |
| `hbond_num_{name}.xvg` | `hbond_analysis` | H-bond count vs time |
| `hbond_matrix_{name}.xpm` | `hbond_analysis` | H-bond existence matrix |
| `hbond_occupancy_{name}.csv` | `hbond_analysis` | Per-pair occupancy table |
| `hbond_occupancy_{name}.png` | `plot_hbond_occupancy` | Occupancy bar chart |
| `hbond_timeseries_{name}.png` | `plot_hbond_occupancy` | H-bond count over time |
| `sasa_total_{name}.xvg` | `sasa_analysis` | Total SASA vs time |
| `sasa_residue_{name}.csv` | `sasa_analysis` | Per-residue mean SASA |
| `sasa_timeseries_{name}.png` | `plot_sasa` | Total SASA over time |
| `sasa_residue_{name}.png` | `plot_sasa` | Per-residue bar chart |
| `covar_{name}.xvg` | `covariance_analysis` | Eigenvalues |
| `eigenvec_{name}.trr` | `covariance_analysis` | Eigenvectors |
| `covar_{name}.dat` | `covariance_analysis` | ASCII covariance matrix |
| `cross_corr_{name}.csv` | `covariance_to_correlation` | DCCM matrix |
| `cross_corr_{name}.jpeg` | `plot_dccm` | DCCM heatmap |
| `network_{name}.png` | `plot_correlation_network` | Weighted network figure |
| `proj_PC{i}_{name}.xvg` | `project_pca` | Per-PC projection time series |
| `pc_projections_{name}.csv` | `project_pca` | Tidy multi-PC CSV |
| `fel_3d_{name}.png` | `plot_free_energy_3d` | 3-D free energy surface |
| `fel_gradient_{name}.png` | `plot_free_energy_3d` | Gradient magnitude figure |
| `bfactor_rmsf_{name}.pdb` | `colour_pdb_by_rmsf` | RMSF-coloured PDB |
| `bfactor_{metric}_{name}.pdb` | `colour_pdb_by_centrality` | Centrality-coloured PDB |

### Checkpoint file

| File | Description |
|---|---|
| `.sim_checkpoint.json` | Pipeline state written to `work_dir`; read by `resume_from_checkpoint()` |

---

## Visualising Results

### PyMOL

```python
# RMSF or centrality-coloured PDB
spectrum b, blue_white_red, minimum=0    # RMSF (low=blue, high=red)
spectrum b, yellow_green_blue            # centrality (low=yellow, high=blue)
```

### ChimeraX

```
open bfactor_rmsf_MyProtein.pdb
color bfactor palette blue:white:red
```

### DCCM — interactive inspection

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("cross_corr_MyProtein.csv")
plt.imshow(data, cmap="bwr", vmin=-1, vmax=1, origin="lower")
plt.colorbar(label="Cross-correlation")
plt.xlabel("Residue")
plt.ylabel("Residue")
plt.tight_layout()
plt.show()
```

### Free energy landscape — custom plotting

The dict returned by `free_energy_landscape()` exposes the raw arrays directly:

```python
land = analysis.free_energy_landscape(pc_data=pc)
# land["G"]          — (M, N) smoothed free energy array (kJ/mol)
# land["grad"]       — (M, N) gradient magnitude array
# land["X"], ["Y"]   — meshgrid of bin centres
# land["basin_pc1"]  — PC1 coordinate of the global minimum
# land["basin_G"]    — free energy at the global minimum
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only pure-Python tests (no GROMACS required)
pytest tests/ -v -m "not integration"

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

The test suite contains 53 tests across 14 classes. Tests requiring GROMACS
(`@requires_gmx`), RDKit (`@requires_rdkit`), or gemmi (`@requires_gemmi`)
are automatically skipped when those tools are not available.

---

## Contributing

Contributions are welcome. Please open an issue to discuss the change before
submitting a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-analysis`
3. Make your changes following the conventions in [CONTRIBUTING.md](CONTRIBUTING.md)
4. Run the test suite: `pytest tests/ -v -m "not integration"`
5. Commit: `git commit -m 'Add salt bridge analysis'`
6. Push: `git push origin feature/my-analysis`
7. Open a pull request

Please follow existing conventions: NumPy docstrings, type hints, `_run()` for
subprocess calls, and update the relevant sections of this README and
`CHANGELOG.md`.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this toolkit in published research, please cite it as:

```bibtex
@software{mdanalysis_gromacs,
  author  = {Sherry, Dean},
  title   = {{MDAnalysis-GROMACS}: An integrated Python toolkit for
             GROMACS simulation preparation and trajectory analysis},
  year    = {2026},
  url     = {https://github.com/DoctorDean/GROMACS-Analysis},
  doi     = {10.5281/zenodo.XXXXXXX},
}
```

Please also cite the underlying tools your analysis depends on:

| Tool | Citation |
|---|---|
| GROMACS | Abraham et al., *SoftwareX* (2015). [doi:10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001) |
| AmberTools | Case et al., *J. Chem. Inf. Model.* (2023). [doi:10.1021/acs.jcim.3c01153](https://doi.org/10.1021/acs.jcim.3c01153) |
| ACPYPE | Sousa da Silva & Vranken, *BMC Res. Notes* (2012). [doi:10.1186/1756-0500-5-367](https://doi.org/10.1186/1756-0500-5-367) |
| NetworkX | Hagberg et al., *Proc. 7th Python in Science Conf.* (2008) |
| Biopython | Cock et al., *Bioinformatics* (2009). [doi:10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163) |
| SciPy | Virtanen et al., *Nature Methods* (2020). [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| RDKit | Landrum, G. *RDKit: Open-source cheminformatics* (2006). [rdkit.org](https://www.rdkit.org) |
| gemmi | Wojdyr, *J. Open Source Softw.* (2022). [doi:10.21105/joss.04200](https://doi.org/10.21105/joss.04200) |
