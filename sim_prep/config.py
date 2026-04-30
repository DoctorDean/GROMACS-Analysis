"""
sim_prep/config.py
==================
YAML / JSON configuration file loader for simulation preparation.

Provides :func:`load_config` which reads a YAML or JSON config file
and returns the appropriate prepper class instantiated and validated,
ready to run.  This lets users drive the entire pipeline from a plain
text file without writing Python.

Config file format
------------------

apo.yaml example::

    type: apo
    protein_name: hsp90
    sim_len: 100
    bx_shp: dodecahedron
    bx_dim: 1.0
    md_name: md_production
    pos_ion: NA
    neg_ion: CL
    work_dir: /data/simulations/hsp90_apo

complex.yaml example::

    type: complex
    protein_name: hsp90_ATP
    ligand_code: ATP
    param_ligand_name: UNL
    remove_ligands:
      - UNL
    sim_len: 100
    bx_shp: dodecahedron
    bx_dim: 1.0
    md_name: md_production
    pos_ion: NA
    neg_ion: CL
    work_dir: /data/simulations/hsp90_ATP
    # Optional: native structure input (Path B)
    # protein_input: 6lu7.cif
    # ligand_input: N3.sdf
    # keep_residues: [ZN]

mixmd.yaml example::

    type: mixmd
    protein_name: hsp90
    sim_len: 100
    bx_shp: dodecahedron
    bx_dim: 1.5
    md_name: md_production
    pos_ion: NA
    neg_ion: CL
    work_dir: /data/simulations/hsp90_mixmd
    ligands:
      - code: ACT
        number: 50
        smiles: "CC(=O)[O-]"
      - code: BNZ
        number: 20
        smiles: "c1ccccc1"

Usage::

    from sim_prep.config import load_config

    sim = load_config("apo.yaml")
    sim.run_all()

    # Or step by step:
    sim = load_config("complex.yaml")
    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from sim_prep.apo     import ApoSimPrepper
from sim_prep.complex import CplxSimPrepper
from sim_prep.mixmd   import MixMDPrepper
from sim_prep.base    import SimulationPrepper

#: Maps the ``type`` field in a config file to its prepper class
_PREPPER_REGISTRY: dict[str, type[SimulationPrepper]] = {
    "apo":     ApoSimPrepper,
    "complex": CplxSimPrepper,
    "mixmd":   MixMDPrepper,
}

# Fields that are handled by the loader itself, not passed to the class
_LOADER_KEYS = {"type"}


def load_config(
    config_path: Union[str, Path],
    run_all: bool = False,
) -> SimulationPrepper:
    """
    Load a YAML or JSON simulation config file and return an instantiated,
    validated prepper object.

    Parameters
    ----------
    config_path : str or Path
        Path to a ``.yaml``, ``.yml``, or ``.json`` config file.
    run_all : bool, optional
        If ``True``, immediately run the full preparation pipeline
        after loading.  Defaults to ``False``.

    Returns
    -------
    SimulationPrepper
        An instantiated subclass (``ApoSimPrepper``, ``CplxSimPrepper``,
        or ``MixMDPrepper``) with ``validate_config()`` and
        ``assign_attributes()`` already called.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If ``type`` is missing or not one of ``"apo"``, ``"complex"``,
        ``"mixmd"``.
    ImportError
        If a YAML file is provided but PyYAML is not installed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to read YAML config files.\n"
                "Install it with:  pip install pyyaml"
            )
        with config_path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

    elif suffix == ".json":
        with config_path.open() as fh:
            raw = json.load(fh)

    else:
        raise ValueError(
            f"Unsupported config format: '{suffix}'.  "
            "Use .yaml, .yml, or .json"
        )

    sim_type = raw.get("type", "").lower()
    if sim_type not in _PREPPER_REGISTRY:
        raise ValueError(
            f"Config 'type' must be one of {list(_PREPPER_REGISTRY)}, "
            f"got: '{sim_type}'"
        )

    # Strip loader-only keys before passing to the class
    kwargs = {k: v for k, v in raw.items() if k not in _LOADER_KEYS}

    prepper_cls = _PREPPER_REGISTRY[sim_type]
    sim = prepper_cls(**kwargs)
    sim.validate_config()
    sim.assign_attributes()

    print(f"✓  Config loaded: {config_path.name}  [{sim_type}]  →  {sim.protein_name}")

    if run_all:
        sim.copy_config_files()
        sim.update_config_files()
        _run_pipeline(sim, raw)

    return sim


def _run_pipeline(sim: SimulationPrepper, raw: dict[str, Any]) -> None:
    """
    Run the full pipeline for the loaded prepper, choosing the correct
    input path for CplxSimPrepper based on config keys present.
    """
    sim_type = raw.get("type", "").lower()

    if sim_type == "apo":
        sim.clean_pdb_file()
        sim.protein_pdb2gmx()
        sim.set_new_box()
        sim.solvate()
        sim.minimise_system()
        sim.nvt_equilibration()
        sim.npt_equilibration()
        sim.production_run()

    elif sim_type == "complex":
        # Determine input path from config keys
        protein_input = raw.get("protein_input")
        ligand_input  = raw.get("ligand_input")
        keep_residues = raw.get("keep_residues", [])

        if protein_input and ligand_input:
            # Path B or C — native / pre-separated structure
            sim.prepare_from_structure(
                protein_input=protein_input,
                ligand_input=ligand_input,
                keep_residues=keep_residues,
            )
        else:
            # Path A — AutoDock complex
            sim.process_autodocked_complex()

        net_charge = raw.get("net_charge")
        atom_type  = raw.get("atom_type", "gaff")
        sim.param_with_amber(
            net_charge=net_charge,
            atom_type=atom_type,
        )
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

    elif sim_type == "mixmd":
        sim.clean_pdb_file()
        sim.protein_pdb2gmx()
        sim.set_new_box()
        net_charges = raw.get("net_charges", {})
        sim.param_all_ligands(net_charges=net_charges)
        sim.top2itp()
        sim.merge_atomtypes()
        sim.build_mixmd()
        sim.solvate()
        sim.minimise_system()
        sim.nvt_equilibration()
        sim.npt_equilibration()
        sim.production_run()


def generate_template(
    sim_type: str,
    output_path: Union[str, Path, None] = None,
) -> str:
    """
    Generate an annotated YAML template for the given simulation type
    and optionally write it to a file.

    Parameters
    ----------
    sim_type : str
        One of ``"apo"``, ``"complex"``, ``"mixmd"``.
    output_path : str or Path, optional
        If provided, write the template to this path.

    Returns
    -------
    str
        The YAML template string.

    Example
    -------
    ::

        from sim_prep.config import generate_template
        generate_template("complex", "my_sim.yaml")
    """
    templates: dict[str, str] = {
        "apo": """\
# Apo protein simulation configuration
# Usage: from sim_prep.config import load_config; sim = load_config("apo.yaml")

type: apo

# ── Required ─────────────────────────────────────────────────────────────────
protein_name: my_protein          # stem of {protein_name}.pdb in work_dir
sim_len: 100                      # production run length (nanoseconds)
bx_shp: dodecahedron              # cubic | dodecahedron | triclinic | octahedron
bx_dim: 1.0                       # box padding in nm (distance from protein edge)
md_name: md_production            # stem for all production output files

# ── Ions ─────────────────────────────────────────────────────────────────────
pos_ion: NA                       # NA | K | MG | CA | ZN
neg_ion: CL

# ── Paths ─────────────────────────────────────────────────────────────────────
work_dir: /path/to/simulation/dir

# ── Optional ──────────────────────────────────────────────────────────────────
# gmx_executable: gmx            # override if gmx is not on PATH
# index_file: index.ndx          # custom .ndx file
""",

        "complex": """\
# Protein–ligand complex simulation configuration
# Usage: from sim_prep.config import load_config; sim = load_config("complex.yaml")

type: complex

# ── Required ─────────────────────────────────────────────────────────────────
protein_name: my_protein_LIG      # stem of the complex PDB (Path A)
                                  # or output file stem (Paths B / C)
ligand_code: LIG                  # three-letter residue code for the ligand
sim_len: 100
bx_shp: dodecahedron
bx_dim: 1.0
md_name: md_production
pos_ion: NA
neg_ion: CL
work_dir: /path/to/simulation/dir

# ── Input path selection ──────────────────────────────────────────────────────
# PATH A — AutoDock complex PDB (default if protein_input / ligand_input absent)
param_ligand_name: UNL            # residue name in the docked PDB (AutoDock = UNL)
remove_ligands:
  - UNL

# PATH B / C — native structure or pre-separated files (uncomment to use)
# protein_input: 6lu7.cif         # .pdb / .cif / .mmcif / .gro
# ligand_input:  N3.sdf           # .pdb / .mol2 / .sdf / .mol / smiles:CC...
# keep_residues: [ZN]             # HETATM residues to retain (e.g. metal ions)

# ── Parameterisation ──────────────────────────────────────────────────────────
# net_charge: -1                  # net formal charge of ligand (auto if omitted)
# atom_type: gaff                 # gaff | gaff2

# ── Optional ──────────────────────────────────────────────────────────────────
# gmx_executable: gmx
# index_file: index.ndx
""",

        "mixmd": """\
# Mixed-solvent (MixMD) simulation configuration
# Usage: from sim_prep.config import load_config; sim = load_config("mixmd.yaml")

type: mixmd

# ── Required ─────────────────────────────────────────────────────────────────
protein_name: my_protein
sim_len: 100
bx_shp: dodecahedron
bx_dim: 1.5                       # larger box recommended for MixMD
md_name: md_production
pos_ion: NA
neg_ion: CL
work_dir: /path/to/simulation/dir

# ── Probe molecules ───────────────────────────────────────────────────────────
# Each probe requires: code (3-letter), number of copies, and SMILES
ligands:
  - code: ACT
    number: 50
    smiles: "CC(=O)[O-]"
  - code: BNZ
    number: 20
    smiles: "c1ccccc1"

# ── Per-ligand net charges (optional) ─────────────────────────────────────────
# net_charges:
#   ACT: -1
#   BNZ: 0

# ── Optional ──────────────────────────────────────────────────────────────────
# gmx_executable: gmx
# index_file: index.ndx
""",
    }

    sim_type = sim_type.lower()
    if sim_type not in templates:
        raise ValueError(
            f"sim_type must be one of {list(templates)}, got: '{sim_type}'"
        )

    content = templates[sim_type]

    if output_path is not None:
        Path(output_path).write_text(content)
        print(f"✓  Template written → {output_path}")

    return content
