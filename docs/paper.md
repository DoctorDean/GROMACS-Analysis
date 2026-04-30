---
title: 'MDAnalysis-GROMACS: An integrated Python toolkit for molecular dynamics simulation preparation and trajectory analysis'

tags:
  - Python
  - molecular dynamics
  - GROMACS
  - structural biology
  - computational chemistry
  - protein simulation
  - free energy landscape
  - principal component analysis

authors:
  - name: [Your Name]
    orcid: 0000-0000-0000-0000
    affiliation: 1

affiliations:
  - name: [Your Institution]
    index: 1

date: 2025

bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulation is a cornerstone technique in structural
biology and computational drug discovery, enabling researchers to observe the
atomic-level motions of proteins, nucleic acids, and small molecules over
timescales from picoseconds to microseconds [@Hospital2015]. Despite the
widespread availability of simulation engines such as GROMACS [@Abraham2015],
the practical workflow from raw protein structure to interpreted simulation
results remains fragmented across multiple command-line tools, scripting
languages, and manual data handling steps. This fragmentation creates a
significant barrier for researchers entering the field and introduces
reproducibility risks through undocumented, ad hoc analysis scripts.

`MDAnalysis-GROMACS` is an open-source Python toolkit that unifies the full
MD pipeline — from initial structure preparation through equilibration,
production simulation, and post-simulation trajectory analysis — into a single
coherent, documented, and testable codebase. The package is composed of two
independent but complementary modules: `sim_prep`, which automates GROMACS
simulation preparation for apo protein, protein–ligand complex, and
mixed-solvent simulation types; and `GromacsAnalysis`, which provides a
high-level interface to trajectory analysis, dynamic cross-correlation,
principal component analysis, free energy landscape computation, and
graph-theoretic network analysis of residue correlation. Together, these
modules reduce a typical protein simulation pipeline from hundreds of manual
command-line invocations to a concise, reproducible Python script.

# Statement of Need

The GROMACS simulation engine is one of the most widely used tools in
computational structural biology [@Abraham2015], yet its native interface is
entirely command-line based and requires users to manually chain together
dozens of commands, interactively select atom groups, manage intermediate
files, and parse custom XVG output formats. Post-simulation analysis is
similarly fragmented: tools such as `gmx rms`, `gmx rmsf`, `gmx covar`,
`gmx hbond`, and `gmx sasa` each have distinct interfaces and output formats,
requiring bespoke parsing code for every analysis type.

Existing Python packages address parts of this problem. MDAnalysis
[@Michaud-Agrawal2011; @Gowers2016] provides a powerful trajectory reading
and atom selection framework, but does not automate GROMACS preparation
pipelines or provide the higher-order analyses (DCCM, FEL, graph networks)
that are standard in structural biology publications. PLUMED [@Tribello2014]
enables enhanced sampling and on-the-fly collective variable computation but
is not a general-purpose analysis toolkit. HTMD [@Doerr2016] and BioSimSpace
[@Hedges2019] offer automation for multiple simulation engines but introduce
substantial dependencies and are oriented towards high-throughput protocols
rather than the single-system, in-depth analysis workflows common in academic
structural biology.

`MDAnalysis-GROMACS` fills the gap between raw GROMACS output and
publication-quality analysis for researchers working with individual protein
systems, particularly in the context of drug discovery, allosteric
communication, and mixed-solvent probe binding site identification. The toolkit
is intentionally scoped to GROMACS and to the analysis methods most commonly
reported in structural biology literature, prioritising depth and correctness
in a well-defined domain over breadth across multiple simulation engines.

# Implementation

## Architecture

The package is divided into two logically independent modules with no
cross-dependency, so either can be used in isolation with existing data.

**`sim_prep`** is an object-oriented preparation package built around an
abstract base class, `SimulationPrepper`, which implements all shared GROMACS
subprocess calls, MDP template management, and the standard preparation
pipeline steps (solvation, energy minimisation, NVT equilibration, NPT
equilibration, and production run). Three concrete subclasses extend this base:

- `ApoSimPrepper` — prepares protein-only (apo) simulations using AMBER99SB-ILDN
  and TIP3P water, covering the complete pipeline from PDB cleaning through to
  production run.
- `CplxSimPrepper` — prepares protein–ligand complex simulations. Ligand
  parameterisation is handled natively in Python via a sequential pipeline of
  Open Babel protonation, ANTECHAMBER AM1-BCC charge assignment, PARMCHK2
  parameter checking, TLEAP topology building, and ACPYPE format conversion
  [@Wang2004; @Sousa2012]. Three input paths are supported: AutoDock Vina
  docked complexes, native PDB/mmCIF structures with separately provided
  ligands, and pre-separated protein and ligand files from tools such as
  Schrödinger Glide. Structure format conversion from mmCIF (including all
  current RCSB PDB and AlphaFold depositions) is provided via `gemmi`
  [@Wojdyr2022], and ligand format conversion from SDF, MOL2, and SMILES
  strings is provided via RDKit [@Landrum2006].
- `MixMDPrepper` — prepares mixed-solvent MD (MixMD) simulations [@Guvench2009],
  in which organic probe molecules are co-solvated with the protein to
  identify cryptic binding sites. Supports multiple simultaneous probe
  molecules with independent parameterisation and `gmx insert-molecules`
  placement.

A `CheckpointMixin` class enables persistent pipeline state so that
preparation runs interrupted by HPC wall-time limits can be resumed from the
point of failure rather than restarted from scratch. A YAML/JSON configuration
loader (`sim_prep.config`) allows entire simulation parameters to be specified
in a plain-text file, improving reproducibility and accessibility for
researchers without extensive Python experience.

**`GromacsAnalysis`** provides a single class encapsulating sixteen analysis
and visualisation methods organised into five conceptual groups:

1. *Essential dynamics* — root mean squared deviation (RMSD), root mean
   squared fluctuation (RMSF), and radius of gyration (Rg) via `gmx rms`,
   `gmx rmsf`, and `gmx gyrate`.
2. *Interaction analysis* — hydrogen bond number time series, donor–acceptor
   pair occupancy, and solvent-accessible surface area (SASA) per residue and
   over time via `gmx hbond` and `gmx sasa`.
3. *Covariance and dimensionality reduction* — covariance matrix
   decomposition, conversion to a dynamic cross-correlation matrix (DCCM)
   [@Ichiye1991], principal component analysis (PCA) via `gmx covar` and
   `gmx anaeig`, and free energy landscape computation by Boltzmann inversion
   of the PC1–PC2 probability density with Gaussian smoothing.
4. *Graph-theoretic network analysis* — construction of a weighted residue
   correlation graph from the DCCM, computation of degree, betweenness,
   closeness, and eigenvector centrality, plus structural reporting of
   k-cores and articulation points.
5. *Structure annotation* — colouring of PDB B-factor columns with RMSF or
   centrality values for direct visualisation in PyMOL or ChimeraX.

All GROMACS subprocess calls are centralised through a single `_run` helper
with uniform error handling. XVG output files are parsed by a general
`_parse_xvg` static method. All analysis outputs are written to the working
directory using consistent naming conventions, and the in-memory return values
of each method can be chained directly into subsequent steps.

## Computational workflow

A complete apo protein simulation analysis pipeline requires approximately
fifteen lines of Python:

```python
from sim_prep import ApoSimPrepper
from gromacs_analysis import GromacsAnalysis

sim = ApoSimPrepper(protein_name="hsp90", sim_len=100,
                    bx_dim=1.0, bx_shp="dodecahedron",
                    md_name="md_production",
                    work_dir="/data/simulations/hsp90_apo")
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

analysis = GromacsAnalysis(md_name="md_production",
                           protein_name="HSP90",
                           work_dir="/data/simulations/hsp90_apo")
analysis.nopbc_and_fit()
analysis.essential_dynamics(time_unit="ns")
analysis.hbond_analysis()
analysis.sasa_analysis()
analysis.covariance_analysis(last=10)
corr = analysis.covariance_to_correlation()
G    = analysis.correlation_network(corr=corr, threshold=0.3)
pc   = analysis.project_pca(first=1, last=2)
land = analysis.free_energy_landscape(pc_data=pc, temperature=300.0)
analysis.plot_free_energy_3d(land)
analysis.colour_pdb_by_centrality(graph=G, metric="betweenness")
```

The equivalent workflow using native GROMACS command-line calls and custom
analysis scripts would require in excess of 200 individual commands and several
hundred lines of ad hoc parsing code, with no standardisation of file naming,
error handling, or output format.

# Availability and Installation

`MDAnalysis-GROMACS` is available at
[https://github.com/DoctorDean/GROMACS-Analysis](https://github.com/DoctorDean/GROMACS-Analysis)
under the MIT licence. The package requires Python ≥ 3.10, GROMACS ≥ 2019,
and the Python dependencies listed in `requirements.txt`
(NumPy [@Harris2020], pandas [@McKinney2010], SciPy [@Virtanen2020],
Matplotlib [@Hunter2007], NetworkX [@Hagberg2008], and Biopython [@Cock2009]).
Ligand parameterisation additionally requires AmberTools [@Case2023],
ACPYPE [@Sousa2012], and Open Babel [@OBoyle2011]. Structure format conversion
requires `gemmi` [@Wojdyr2022] and RDKit [@Landrum2006]. Full installation
instructions, including a conda environment specification for ligand
parameterisation dependencies, are provided in `LIGAND_PARAMETERISATION.md`.

# References
