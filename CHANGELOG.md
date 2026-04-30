# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [1.2.0] – 2026

### Added — analysis (`gromacs_analysis.py`)
- `hbond_analysis()` — hydrogen bond number time series and per-donor–acceptor-pair
  occupancy via `gmx hbond`; configurable distance/angle cutoffs and time window;
  outputs XVG time series, XPM existence matrix, and a tidy occupancy CSV
- `_parse_hbond_matrix()` — internal parser for GROMACS XPM H-bond existence
  matrices; produces per-pair occupancy fraction from raw binary frame data
- `plot_hbond_occupancy()` — two-panel figure: horizontal bar chart of top-N
  donor–acceptor pairs by occupancy and a time-series of H-bond count with
  rolling mean overlay
- `sasa_analysis()` — total and per-residue solvent-accessible surface area via
  `gmx sasa`; configurable probe radius and time window; exports per-residue
  SASA as tidy CSV alongside XVG outputs
- `plot_sasa()` — two-panel figure: total SASA over time with rolling mean and
  mean reference line, and a per-residue bar chart colour-mapped by exposure level

### Added — simulation preparation (`sim_prep/`)
- `sim_prep/checkpoint.py` — `CheckpointMixin` class providing persistent pipeline
  state via a JSON checkpoint file (`.sim_checkpoint.json`) written to the working
  directory
  - `resume_from_checkpoint()` — loads prior state; all subsequent pipeline calls
    silently skip already-completed steps
  - `run_step(name, fn)` — manually guard any callable with checkpoint logic
    without requiring the decorator
  - `@checkpoint_step` decorator — wraps any pipeline method to skip execution
    if already marked complete
  - `reset_checkpoint(steps=None)` — un-mark specific steps or fully clear state
  - `checkpoint_status()` — print a summary of completed and pending steps
- `sim_prep/config.py` — YAML/JSON configuration file loader
  - `load_config(path)` — reads a `.yaml`, `.yml`, or `.json` config file and
    returns a fully instantiated, validated prepper object ready to run
  - `generate_template(sim_type, output_path)` — writes an annotated YAML
    template for `"apo"`, `"complex"`, or `"mixmd"` simulation types with
    inline documentation of every parameter
  - `run_all=True` flag on `load_config()` — drives the full pipeline from a
    config file with a single function call, routing to the correct input path
    for `CplxSimPrepper` based on keys present in the config

### Added — configuration templates (`config/examples/`)
- `config/examples/apo.yaml` — annotated reference config for apo simulations
- `config/examples/complex.yaml` — annotated reference config for protein–ligand
  complex simulations; documents all three input paths (AutoDock, native
  structure, pre-separated files) with inline comments
- `config/examples/mixmd.yaml` — annotated reference config for mixed-solvent
  simulations with multi-probe ligand definitions and per-ligand charge settings

### Added — packaging
- `pyproject.toml` — PEP 517/518 build configuration; makes the package
  installable with `pip install .`; defines core and optional dependency groups
  (`[structure]`, `[dev]`, `[docs]`); includes project metadata, classifiers,
  and entry-point definitions

### Added — tests (`tests/test_suite.py`)
- 53 tests across 14 test classes covering:
  - `TestParseXvg` (4 tests) — XVG parsing correctness and comment-line skipping
  - `TestCleanPdb` (6 tests) — HETATM removal, water stripping, ANISOU removal,
    keep_residues override, altLoc handling
  - `TestPrepareStructureRouting` (5 tests + 2 optional) — format dispatch, error
    handling, SMILES generation (RDKit), mmCIF conversion (gemmi)
  - `TestSelectionGroups` (2 tests) — default values and custom override
  - `TestFreeEnergyLandscape` (4 tests) — return keys, vmax normalisation, array
    shape consistency, basin within PC range
  - `TestCovarianceToCorrelation` (2 tests) — diagonal = 1.0, values in [-1, 1]
  - `TestCorrelationNetwork` (3 tests) — graph type, no self-loops, threshold
    effect on edge count
  - `TestApoValidation` (5 tests) — all validation failure modes
  - `TestComplexValidation` (2 tests) — ligand code required, valid config passes
  - `TestMixMDValidation` (2 tests) — empty ligands, missing ligand fields
  - `TestConfigLoader` (7 tests) — YAML load, JSON load, invalid type, missing
    file, template generation, invalid template type
  - `TestCheckpoint` (7 tests) — no-checkpoint state, persist on mark, resume
    across instances, skip completed, execute new, full reset, partial reset
  - `TestAmberDependencies` (2 tests) — return type, monkeypatched missing tools
  - `TestGromacsIntegration` (1 test, `@requires_gmx`) — GROMACS version check

### Added — manuscript (`paper.md`, `paper.bib`)
- JOSS-format manuscript with Summary, Statement of Need, Implementation,
  and Availability sections
- `paper.bib` — 22 BibTeX entries covering all cited software and methods

### Changed
- `gromacs_analysis.py` workflow docstring updated to reflect 16 pipeline steps
  (was 12); step numbering adjusted throughout
- `requirements.txt` restructured into three tiers: core (always required),
  structure handling (gemmi, rdkit), and optional (pdbfixer); added PyYAML
  to core tier

---

## [1.1.0] – 2026

### Added — analysis (`gromacs_analysis.py`)
- `project_pca()` — projects the fitted trajectory onto N principal components
  via `gmx anaeig`; exports tidy per-PC CSV; defaults to PC1 and PC2
- `free_energy_landscape()` — 2-D FEL from PC projections via Boltzmann
  inversion; Gaussian smoothing; gradient magnitude computation; basin
  detection
- `plot_free_energy_3d()` — 3-D free energy surface and gradient magnitude
  figure (3-D surface + 2-D heatmap with contour overlay)
- `colour_pdb_by_rmsf()` — writes per-residue RMSF from XVG into PDB B-factor
  column using Biopython; ready for PyMOL/ChimeraX visualisation
- `colour_pdb_by_centrality()` — writes any of four centrality metrics (degree,
  betweenness, closeness, eigenvector) into PDB B-factor column
- `_apply_bfactors()` — internal static method for bulk B-factor assignment
- `_save_pdb()` — internal PDBIO wrapper
- `_parse_xvg()` — general two-column XVG parser (replaces ad hoc per-method
  parsing)

### Added — simulation preparation (`sim_prep/`)
- `sim_prep/` package with abstract base class and three concrete subclasses
- `sim_prep/base.py` — `SimulationPrepper` ABC with shared pipeline steps:
  `clean_pdb_file()`, `protein_pdb2gmx()`, `set_new_box()`, `solvate()`,
  `minimise_system()`, `nvt_equilibration()`, `npt_equilibration()`,
  `production_run()`, `update_topology_molecules()`
- `sim_prep/apo.py` — `ApoSimPrepper` for protein-only simulations
- `sim_prep/complex.py` — `CplxSimPrepper` for protein–ligand complex
  simulations; three input paths (AutoDock, native structure, pre-separated);
  `process_autodocked_complex()`, `prepare_from_structure()`,
  `prepare_from_separate_files()`, `param_with_amber()`, `gro2itp()`,
  `generate_ligand_ndx()`, `build_gmx_complex()`, `build_complex_topology()`,
  `make_prot_lig_ndx()`
- `sim_prep/mixmd.py` — `MixMDPrepper` for mixed-solvent simulations;
  `param_all_ligands()`, `top2itp()`, `merge_atomtypes()`, `build_mixmd()`
- `SelectionGroups` dataclass for GROMACS group index configuration

### Added — utilities (`utils/`)
- `utils/amber_params.py` — `AmberParameteriser` class; native Python
  replacement for the former `param_with_amber.sh` shell script; exposes
  `net_charge`, `atom_type`, `charge_method`, and `verbose` parameters;
  `check_dependencies()` standalone function for pre-flight tool verification
- `utils/structure_io.py` — `prepare_structure()` function supporting PDB,
  mmCIF, GRO, MOL2, SDF, and SMILES input; `_clean_pdb()`, `_cif_to_pdb()`,
  `_gro_to_pdb()`, `_mol2_to_pdb()`, `_sdf_to_pdb()`, `_smiles_to_pdb()`

### Added — examples and documentation
- `examples/apo_example.py`, `examples/complex_example.py` (three input paths),
  `examples/mixmd_example.py`
- `LIGAND_PARAMETERISATION.md` — full AmberTools + ACPYPE setup guide including
  conda environment instructions, HPC usage, input requirements, output file
  descriptions, and common error fixes
- `STRUCTURE.md` — repo layout reference with design principle documentation

### Removed
- `utils/param_with_amber.sh` — replaced by `utils/amber_params.py`

---

## [1.0.0] – 2025

### Added
- `GromacsAnalysis` class with unified interface for all trajectory analyses
- `SelectionGroups` dataclass for flexible GROMACS group configuration
- `nopbc_and_fit()` — PBC removal and rot+trans fitting via `gmx trjconv`
- `essential_dynamics()` — RMSD, RMSF, and radius of gyration
- `covariance_analysis()` — full `gmx covar` wrapper with all flags exposed
- `covariance_to_correlation()` — converts ASCII covariance matrix to per-residue DCCM
- `plot_dccm()` — publication-quality DCCM heatmap
- `correlation_network()` — weighted NetworkX graph from DCCM with 10 graph metrics
- `plot_correlation_network()` — spring-layout weighted network figure
- `README.md`, `requirements.txt`, `LICENSE` (MIT), `.gitignore`,
  `CHANGELOG.md`, `CONTRIBUTING.md`
