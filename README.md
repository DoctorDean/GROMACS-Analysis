# MDAnalysis-GROMACS

A Python toolkit for post-simulation analysis of GROMACS molecular dynamics trajectories. Wraps common `gmx` commands into a single, chainable class and extends them with pure-Python analyses — dynamic cross-correlation, free energy landscapes, correlation networks, and structure colouring — so a complete analysis pipeline runs in one script.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Pipeline](#full-pipeline)
- [API Reference](#api-reference)
- [Output Files](#output-files)
- [Visualising Results](#visualising-results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

| Stage | Method | Description |
|---|---|---|
| Trajectory prep | `nopbc_and_fit()` | Remove PBC, least-squares fit |
| Essential dynamics | `essential_dynamics()` | RMSD, RMSF, radius of gyration |
| PCA / covariance | `covariance_analysis()` | `gmx covar` wrapper |
| DCCM | `covariance_to_correlation()` | Covariance → normalised cross-correlation matrix |
| DCCM plot | `plot_dccm()` | Heatmap with configurable colourmap |
| Correlation network | `correlation_network()` | Weighted graph + 10 network metrics |
| Network plot | `plot_correlation_network()` | Weighted spring-layout graph |
| PC projections | `project_pca()` | `gmx anaeig` → tidy CSV |
| Free energy landscape | `free_energy_landscape()` | 2-D FEL via Boltzmann inversion |
| FEL plot | `plot_free_energy_3d()` | 3-D surface + gradient magnitude |
| PDB colouring | `colour_pdb_by_rmsf()` | B-factor ← RMSF |
| PDB colouring | `colour_pdb_by_centrality()` | B-factor ← degree / betweenness / closeness / eigenvector centrality |

**Flexible by design.** Every selection group, file path, time window, colourmap, and analysis parameter is overridable. Works with any protein-in-water or custom system — pass a `.ndx` index file and override `SelectionGroups` for membrane proteins, multi-chain complexes, or ligand-bound systems.

---

## Requirements

### External

- [GROMACS](https://www.gromacs.org/) ≥ 2019 (the `gmx` executable must be on your `PATH`, or pass the full path via `gmx_executable`)

### Python

- Python ≥ 3.10
- See [`requirements.txt`](requirements.txt) for the full list

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mdanalysis-gromacs.git
cd mdanalysis-gromacs
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GROMACS is not installed via pip.** Follow the [official GROMACS installation guide](https://manual.gromacs.org/documentation/current/install-guide/index.html) or install via your system package manager / HPC module system.

---

## Quick Start

```python
from gromacs_analysis import GromacsAnalysis

sim = GromacsAnalysis(
    md_name="md_production",   # stem of your .tpr / .xtc files
    protein_name="MyProtein",  # used to label all output files
    work_dir="/path/to/sim",   # directory containing simulation files
)

sim.nopbc_and_fit()
sim.essential_dynamics()
```

This produces `md_production_noPBC.xtc`, `md_production_fitted.xtc`, and the standard RMSD / RMSF / Rg XVG files in the working directory.

---

## Full Pipeline

### Setup

```python
from gromacs_analysis import GromacsAnalysis, SelectionGroups

# Override selection groups if using a custom .ndx file
groups = SelectionGroups(
    center="1",          # Protein  – centre for PBC removal
    output="0",          # System   – atoms written to output trajectory
    fit="4",             # Backbone – used for rot+trans fitting
    rmsf="4",            # Backbone – RMSF per residue
    rg="1",              # Protein  – radius of gyration
    covar_fit="4",       # Backbone – fit reference for gmx covar
    covar_analysis="4",  # Backbone – atoms decomposed in PCA
)

sim = GromacsAnalysis(
    md_name="md_production",
    protein_name="HSP90",
    work_dir="/data/simulations/hsp90_apo",
    groups=groups,
    index_file="index.ndx",   # optional; omit for standard protein-in-water
)
```

### Trajectory preparation

```python
sim.nopbc_and_fit()
```

### Essential dynamics (RMSD, RMSF, Rg)

```python
sim.essential_dynamics(time_unit="ns")
```

### Covariance / PCA

```python
sim.covariance_analysis(
    time_unit="ns",
    fit=True,
    use_pbc=False,
    last=10,           # write first 10 eigenvectors only
)
```

### Dynamic cross-correlation matrix (DCCM)

```python
corr = sim.covariance_to_correlation()
sim.plot_dccm(corr=corr, cmap="bwr")
```

### Correlation network

```python
G = sim.correlation_network(corr=corr, threshold=0.3)
sim.plot_correlation_network(G)
```

### Free energy landscape

```python
# Project trajectory onto PC1 and PC2
pc_data = sim.project_pca(first=1, last=2, time_unit="ns")

# Compute 2-D FEL at 300 K
landscape = sim.free_energy_landscape(
    pc_data=pc_data,
    bin_width=2.0,
    sigma=1.5,
    temperature=300.0,
)

# 3-D surface + gradient magnitude figures
sim.plot_free_energy_3d(landscape, cmap_G="viridis", cmap_grad="plasma")
```

### PDB colouring

```python
# B-factor ← per-residue RMSF
sim.colour_pdb_by_rmsf(pdb_file="average_structure.pdb")

# B-factor ← network centrality (one PDB per metric)
for metric in ["degree", "betweenness", "closeness", "eigenvector"]:
    sim.colour_pdb_by_centrality(
        pdb_file="average_structure.pdb",
        graph=G,
        metric=metric,
    )
```

### Comparing two systems

Each `GromacsAnalysis` instance is self-contained, so side-by-side comparisons are straightforward:

```python
apo  = GromacsAnalysis("md_production", "HSP90_apo",  work_dir="/data/apo")
holo = GromacsAnalysis("md_production", "HSP90_holo", work_dir="/data/holo")

for sim in [apo, holo]:
    sim.nopbc_and_fit()
    sim.essential_dynamics()
    sim.covariance_analysis()
    corr = sim.covariance_to_correlation()
    G    = sim.correlation_network(corr=corr)
    pc   = sim.project_pca()
    land = sim.free_energy_landscape(pc_data=pc)
    sim.plot_free_energy_3d(land)
```

---

## API Reference

### `SelectionGroups`

Dataclass holding GROMACS selection group indices for each analysis step. All fields default to standard protein-in-water indices (`0` = System, `1` = Protein, `4` = Backbone).

| Field | Default | Used by |
|---|---|---|
| `center` | `"1"` | `nopbc_and_fit` — centering group |
| `output` | `"0"` | `nopbc_and_fit` — output group |
| `fit` | `"1"` | `nopbc_and_fit`, `essential_dynamics` |
| `rmsf` | `"4"` | `essential_dynamics` |
| `rg` | `"1"` | `essential_dynamics` |
| `covar_fit` | `"4"` | `covariance_analysis` |
| `covar_analysis` | `"4"` | `covariance_analysis` |

### `GromacsAnalysis(md_name, protein_name, work_dir, groups, index_file, gmx_executable)`

Main analysis class. All methods write output files to `work_dir` and return data objects for in-memory chaining.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `md_name` | `str` | — | Stem of `.tpr` / `.xtc` files |
| `protein_name` | `str` | — | Label used in all output filenames |
| `work_dir` | `str \| Path` | `cwd` | Directory with simulation files |
| `groups` | `SelectionGroups` | default groups | GROMACS selection indices |
| `index_file` | `str` | `None` | Path to custom `.ndx` |
| `gmx_executable` | `str` | `"gmx"` | GROMACS binary name or full path |

#### Key method signatures

```
nopbc_and_fit()
essential_dynamics(time_unit="ns")
covariance_analysis(begin, end, dt, time_unit, fit, use_pbc, mass_weighted, last, index_file)
covariance_to_correlation(dat_file) → np.ndarray
plot_dccm(corr, csv_file, cmap, vmin, vmax, major_tick_interval, ...)
correlation_network(corr, csv_file, threshold) → nx.Graph
plot_correlation_network(G, title, figsize, node_size, ...)
project_pca(first, last, begin, end, dt, skip, time_unit, ...) → dict
free_energy_landscape(pc_data, csv_file, pc_x, pc_y, bin_width, sigma, temperature, kB) → dict
plot_free_energy_3d(landscape, cmap_G, cmap_grad, elev, azim, ...)
colour_pdb_by_rmsf(pdb_file, xvg_file, output_file, default_bfactor) → Path
colour_pdb_by_centrality(pdb_file, graph, metric, output_file, residue_offset, ...) → Path
```

Full docstrings with parameter descriptions are available in the source or via `help(GromacsAnalysis.<method>)`.

---

## Output Files

| File | Produced by | Description |
|---|---|---|
| `{md}_noPBC.xtc` | `nopbc_and_fit` | PBC-corrected trajectory |
| `{md}_fitted.xtc` | `nopbc_and_fit` | Fitted trajectory |
| `rmsd_{name}.xvg` | `essential_dynamics` | Backbone RMSD vs time |
| `rmsf_{name}.xvg` | `essential_dynamics` | Per-residue RMSF |
| `rg_{name}.xvg` | `essential_dynamics` | Radius of gyration vs time |
| `average_structure_{name}.pdb` | `essential_dynamics` | Time-averaged structure |
| `covar_{name}.xvg` | `covariance_analysis` | Eigenvalues |
| `eigenvec_{name}.trr` | `covariance_analysis` | Eigenvectors |
| `covar_{name}.dat` | `covariance_analysis` | ASCII covariance matrix |
| `cross_corr_{name}.csv` | `covariance_to_correlation` | DCCM matrix |
| `cross_corr_{name}.jpeg` | `plot_dccm` | DCCM heatmap figure |
| `network_{name}.png` | `plot_correlation_network` | Weighted network figure |
| `proj_PC{i}_{name}.xvg` | `project_pca` | Per-PC projection time series |
| `pc_projections_{name}.csv` | `project_pca` | Tidy multi-PC CSV |
| `fel_3d_{name}.png` | `plot_free_energy_3d` | 3-D free energy surface |
| `fel_gradient_{name}.png` | `plot_free_energy_3d` | Gradient magnitude figure |
| `bfactor_rmsf_{name}.pdb` | `colour_pdb_by_rmsf` | RMSF-coloured PDB |
| `bfactor_{metric}_{name}.pdb` | `colour_pdb_by_centrality` | Centrality-coloured PDB |

---

## Visualising Results

### RMSF / centrality PDB colouring in PyMOL

```python
# Open the output PDB in PyMOL, then:
spectrum b, blue_white_red, minimum=0         # RMSF
spectrum b, yellow_green_blue                 # centrality
```

### RMSF / centrality PDB colouring in ChimeraX

```
open bfactor_rmsf_MyProtein.pdb
color bfactor palette blue:white:red
```

### DCCM

The JPEG heatmap is ready to use directly. For interactive inspection, load `cross_corr_{name}.csv` into Python:

```python
import numpy as np, matplotlib.pyplot as plt
data = np.loadtxt("cross_corr_MyProtein.csv")
plt.imshow(data, cmap="bwr", vmin=-1, vmax=1, origin="lower")
plt.colorbar()
plt.show()
```

### Free energy landscape

The landscape dictionary returned by `free_energy_landscape()` contains the raw `G`, `grad`, `X`, `Y` arrays for custom downstream plotting.

---

## Contributing

Contributions are welcome. Please open an issue to discuss the change before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-analysis`)
3. Commit your changes (`git commit -m 'Add hydrogen bond analysis'`)
4. Push to the branch (`git push origin feature/my-analysis`)
5. Open a pull request

Please follow the existing code style (NumPy docstrings, type hints, `_run()` for subprocess calls) and add a brief description to the relevant section of this README.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this toolkit in published research, please cite the underlying tools:

- **GROMACS**: Abraham et al., *SoftwareX* (2015). https://doi.org/10.1016/j.softx.2015.06.001
- **NetworkX**: Hagberg et al., *Proceedings of the 7th Python in Science Conference* (2008).
- **Biopython**: Cock et al., *Bioinformatics* (2009). https://doi.org/10.1093/bioinformatics/btp163
- **SciPy**: Virtanen et al., *Nature Methods* (2020). https://doi.org/10.1038/s41592-019-0686-2

A citation for this repository itself (BibTeX placeholder):

```bibtex
@software{mdanalysis_gromacs,
  author  = Dean Sherry,
  title   = {MDAnalysis-GROMACS: A Python Toolkit for GROMACS Post-Simulation Analysis},
  year    = {2025},
  url     = {https://github.com/DoctorDean/mdanalysis-gromacs},
}
```
