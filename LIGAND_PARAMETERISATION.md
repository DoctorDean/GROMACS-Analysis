# Ligand Parameterisation: AmberTools + ACPYPE Setup Guide

This guide covers everything needed to run `CplxSimPrepper.param_with_amber()` and `MixMDPrepper.param_all_ligands()` — the pipeline steps that convert a ligand PDB file into GROMACS-compatible force-field topology.

---

## Overview

The parameterisation pipeline runs five external tools in sequence:

```
{ligand}.pdb
    │
    ▼  Open Babel (obabel)
{ligand}_h.pdb          ← protonated PDB
    │
    ▼  antechamber (AmberTools)
{ligand}.mol2           ← AM1-BCC partial charges + GAFF atom types
    │
    ▼  parmchk2 (AmberTools)
{ligand}.frcmod         ← missing/supplemental GAFF parameters
    │
    ▼  tleap (AmberTools)
{ligand}.prmtop         ← Amber topology
{ligand}.rst7           ← Amber coordinates
    │
    ▼  ACPYPE
{ligand}.amb2gmx/
    ├── {ligand}_GMX.top      ← GROMACS topology
    ├── {ligand}_GMX.gro      ← GROMACS structure
    └── posre_{ligand}.itp    ← position restraints
```

All five tools must be installed and available on your `PATH` before calling parameterisation methods.

---

## Requirements

| Tool | Minimum version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| [AmberTools](https://ambermd.org/AmberTools.php) | ≥ 23 | `antechamber`, `parmchk2`, `tleap` |
| [ACPYPE](https://github.com/alanwilter/acpype) | ≥ 2022.7.21 | Amber → GROMACS conversion |
| [Open Babel](https://openbabel.org) | ≥ 3.1 | Ligand protonation |

> **Note**: AmberTools is free. The full Amber package (which includes GPU-accelerated MD) requires a licence, but AmberTools alone is sufficient for parameterisation.

---

## Installation

### Recommended: conda environment

A self-contained conda environment is the most reliable way to install all three tools without conflicting with system packages or an existing GROMACS environment.

```bash
conda create -n ambertools python=3.10 -y
conda activate ambertools
```

Install all three tools in one command:

```bash
conda install -c conda-forge ambertools acpype openbabel -y
```

Verify the installation:

```bash
antechamber -h     # should print AmberTools antechamber help
parmchk2 -h        # should print parmchk2 help
tleap -h           # should print tleap help
acpype --version   # should print ACPYPE version
obabel --version   # should print Open Babel version
```

If any command prints `command not found`, the conda environment is not active or the tool did not install correctly. Try:

```bash
conda install -c conda-forge <tool-name> -y
```

### Alternative: pip (ACPYPE only)

ACPYPE is also available via pip, which can be useful if you want to use a system AmberTools installation:

```bash
pip install acpype
```

> AmberTools and Open Babel **cannot** be installed via pip on most systems. Use conda for these.

---

## Activating the environment before running Python

The conda environment must be active when you launch Python. The simplest approach is to activate it in your terminal before running scripts:

```bash
conda activate ambertools
python apo_example.py
```

### Running from a different environment

If your main workflow runs in a different Python environment (e.g. a GROMACS analysis environment), you have two options:

**Option 1 — Install all tools into your main environment:**
```bash
conda activate your-main-env
conda install -c conda-forge ambertools acpype openbabel -y
```

**Option 2 — Call the `check_dependencies()` function early** to confirm all tools are visible before starting a long simulation pipeline:

```python
from utils.amber_params import check_dependencies

missing = check_dependencies()
if missing:
    raise EnvironmentError(
        f"Missing executables: {missing}\n"
        "Activate the ambertools conda environment before running."
    )
```

---

## HPC / cluster usage

On HPC systems, AmberTools is often available as a module rather than a conda package.

```bash
module load ambertools/23
module load openbabel/3.1.1
# ACPYPE is usually not available as a module — install it in a user environment:
pip install --user acpype
```

Verify tool availability in your job script before the Python call:

```bash
#!/bin/bash
#SBATCH --job-name=md_prep
#SBATCH ...

module load ambertools/23
module load openbabel/3.1.1
export PATH="$HOME/.local/bin:$PATH"  # for pip-installed acpype

python complex_example.py
```

---

## Usage

### Single ligand (protein–ligand complex)

```python
from sim_prep.complex import CplxSimPrepper

sim = CplxSimPrepper(
    protein_name="hsp90_ATP",
    ligand_code="ATP",
    ...
)
sim.process_autodocked_complex()

# Default: AM1-BCC charges, GAFF atom types
sim.param_with_amber()

# Charged ligand (e.g. phosphate group with net charge -2):
sim.param_with_amber(net_charge=-2)

# GAFF2 atom types for better accuracy with newer GAFF2-specific parameters:
sim.param_with_amber(atom_type="gaff2")
```

### Multiple probes (MixMD)

```python
from sim_prep.mixmd import MixMDPrepper

sim = MixMDPrepper(
    protein_name="hsp90",
    ligands=[
        {"code": "ACT", "number": 50, "smiles": "CC(=O)[O-]"},
        {"code": "BNZ", "number": 20, "smiles": "c1ccccc1"},
    ],
    ...
)

# Parameterise all probes; specify charges for non-neutral molecules
sim.param_all_ligands(
    net_charges={"ACT": -1, "BNZ": 0},
)
```

---

## Input PDB requirements

The input file `{ligand_code}.pdb` must:

- Contain **only the ligand** — no protein atoms, no water, no counter-ions.
- Use the ligand code as the residue name in columns 18–20 of each ATOM/HETATM record.
- Have **no hydrogen atoms** — Open Babel adds these in step 1. If you pre-add hydrogens, pass `-h` to `obabel` manually and note that it will add hydrogens again (producing duplicates). In that case, use a pre-protonated PDB and set `charge_method="mul"` or `charge_method="gas"`.
- Be in standard PDB format with correct element symbols in column 77–78.

> **AutoDock Vina users**: `process_autodocked_complex()` extracts the ligand automatically from the docked complex PDB and writes `{ligand_code}.pdb` ready for parameterisation. You do not need to prepare this file manually.

---

## Understanding the output files

After a successful run, the `{ligand_code}.amb2gmx/` directory contains:

| File | Description |
|---|---|
| `{ligand_code}_GMX.top` | Full GROMACS topology including `[ atomtypes ]`, `[ moleculetype ]`, and `[ atoms ]` sections |
| `{ligand_code}_GMX.gro` | GROMACS structure file (coordinates) |
| `posre_{ligand_code}.itp` | Position-restraint ITP for the ligand heavy atoms, used during NVT/NPT equilibration |

Intermediate files written to the working directory:

| File | Description |
|---|---|
| `{ligand_code}_h.pdb` | Protonated input PDB |
| `{ligand_code}.mol2` | MOL2 file with AM1-BCC partial charges |
| `{ligand_code}.frcmod` | Supplemental GAFF parameters |
| `{ligand_code}.prmtop` | Amber topology |
| `{ligand_code}.rst7` | Amber coordinates |
| `sqm.out` | SQM semi-empirical QM output — inspect this if antechamber fails |
| `tleap.in` | tleap input script |

---

## Common problems and fixes

### `antechamber` fails with "Calculation may not have completed"

The AM1-BCC SQM job did not converge. Check `sqm.out` for the error. Common causes:

- **Incorrect net charge**: The most common cause. If the molecule carries a formal charge (e.g. phosphate = -1 per group, carboxylate = -1, protonated amine = +1), set `net_charge` explicitly: `sim.param_with_amber(net_charge=-2)`.
- **Radical / unusual spin state**: Set `multiplicity=2` for radicals.
- **Very large ligand** (> 100 heavy atoms): The SQM calculation may time out. Consider using `charge_method="gas"` (Gasteiger, no QM) as a fallback, though accuracy will be lower.

### `parmchk2` writes `ATTN, need revision` comments in the FRCMOD file

These indicate GAFF parameters that could not be estimated reliably from analogies. The simulation will still run, but torsion/angle energetics for those terms may be inaccurate. Options:

- Switch to `atom_type="gaff2"` — GAFF2 has broader coverage of drug-like fragments.
- Use QM-derived torsion parameters from a Gaussian/ORCA scan fitted with `frcmod` tools.
- Consult the [GAFF2 paper](https://doi.org/10.1021/acs.jctc.5b00255) for guidance on which functional groups are well-covered.

### ACPYPE fails with `KeyError` or produces empty files

Usually caused by the Amber topology not being written correctly by tleap. Check:

```bash
cat sqm.out | tail -20
cat leap.log
```

The most common root cause is again an incorrect net charge feeding through to an inconsistent MOL2 file.

### `obabel: command not found`

Open Babel is not on your PATH. If using conda:

```bash
conda activate ambertools
which obabel   # should now resolve
```

If not found after activation:

```bash
conda install -c conda-forge openbabel -y
```

### Residue name mismatch after ACPYPE

ACPYPE names the ligand residue in the output topology using the residue name it finds in the input MOL2 file. If `antechamber` has rewritten the residue name to `UNL` or `UNK` (a known ACPYPE behaviour for some molecule types), the topology residue name will not match your `ligand_code`. The `AmberParameteriser` class uses your ligand code as the input file name, which usually propagates correctly — but if you see mismatches, inspect `{ligand_code}.mol2` and check that the residue name in the `@<TRIPOS>ATOM` section matches your `ligand_code`.

---

## Verifying a parameterisation before running a full simulation

Before committing to a multi-day production run, it is worth doing a short (1–2 ns) test simulation and checking:

1. **No `nan` energies in `em.log`** — indicates a topology error or clashing atoms.
2. **RMSD stays bounded after equilibration** — the ligand should not drift out of the binding site.
3. **Inspect the FRCMOD file** — any `ATTN` entries should be reviewed before production.

```python
# Quick energy minimisation only, to check the topology is valid:
sim.solvate()
sim.minimise_system()
# Open em.log and check: "Potential Energy" should be large-negative, not nan/inf
```
