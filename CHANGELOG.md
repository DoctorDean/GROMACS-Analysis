# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [1.0.0] – 2025

### Added
- `GromacsAnalysis` class with unified interface for all analyses
- `SelectionGroups` dataclass for flexible GROMACS group configuration
- `nopbc_and_fit()` — PBC removal and rot+trans fitting via `gmx trjconv`
- `essential_dynamics()` — RMSD, RMSF, and radius of gyration via `gmx rms`, `gmx rmsf`, `gmx gyrate`
- `covariance_analysis()` — full `gmx covar` wrapper with all flags exposed
- `covariance_to_correlation()` — converts ASCII covariance matrix to a per-residue DCCM
- `plot_dccm()` — publication-quality DCCM heatmap with configurable colourmap, ticks, and font
- `correlation_network()` — builds a weighted NetworkX graph from the DCCM and prints 10 graph-theoretic metrics
- `plot_correlation_network()` — spring-layout network figure coloured by degree centrality
- `project_pca()` — projects the fitted trajectory onto N principal components via `gmx anaeig`; exports tidy CSV
- `free_energy_landscape()` — 2-D FEL via Boltzmann inversion with Gaussian smoothing and gradient magnitude
- `plot_free_energy_3d()` — 3-D free energy surface + 2-D gradient magnitude heatmap with contour overlay
- `colour_pdb_by_rmsf()` — writes per-residue RMSF into PDB B-factor column
- `colour_pdb_by_centrality()` — writes any of four centrality metrics into PDB B-factor column
- `_parse_xvg()` static helper for reading two-column GROMACS XVG files
- `requirements.txt`, `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`
