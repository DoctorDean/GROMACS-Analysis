# Contributing to MDAnalysis-GROMACS

Thank you for considering a contribution. This document covers how to set up a development environment, the code conventions used throughout the project, and the pull request process.

---

## Getting started

### 1. Fork and clone

```bash
git clone https://github.com/<your-fork>/GROMACS-Analysis.git
cd GROMACS-Analysis
```

### 2. Create a development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create a feature branch

```bash
git checkout -b feature/my-new-analysis
```

---

## Code conventions

### Style
- Follow [PEP 8](https://peps.python.org/pep-0008/).  Line length ≤ 99 characters.
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public methods and classes.
- Add type hints to all function signatures.

### Adding a new analysis method

1. Add the method to `GromacsAnalysis` in `gromacs_analysis.py`.
2. If it wraps a `gmx` command, build the command list and call `self._run(command, stdin, label)`.
3. If it parses XVG output, use or extend `self._parse_xvg()`.
4. Use `self.work_dir / f"output_{self.protein_name}.ext"` for all default output paths.
5. Update the workflow list in the module docstring (lines 10–21).
6. Update the `__main__` demo block with a minimal usage example.
7. Add the new method to the API Reference table and Output Files table in `README.md`.
8. Add an entry to `CHANGELOG.md` under `[Unreleased]`.

### Subprocess calls

All GROMACS subprocess calls must go through `self._run()`.  Do not call `subprocess.run()` directly in analysis methods.  This ensures consistent error handling and output formatting.

---

## Pull request checklist

- [ ] Feature branch is up to date with `main`
- [ ] Code follows the style conventions above
- [ ] All new public methods have NumPy docstrings with Parameters, Returns, and Output files sections
- [ ] `README.md` tables updated
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No simulation data files (`.xtc`, `.tpr`, `.xvg`, etc.) committed

---

## Reporting bugs

Open a [GitHub Issue](https://github.com/DoctorDean/GROMACS-Analysis/issues) with:

- Your GROMACS version (`gmx --version`)
- Your Python version (`python --version`)
- The full traceback
- A minimal description of the system (protein-in-water, membrane, multi-chain, etc.)
