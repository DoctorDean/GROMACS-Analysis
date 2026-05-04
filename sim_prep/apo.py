"""
apo.py
======
Simulation preparation for apo (protein-only) systems.

The ``ApoSimPrepper`` class inherits all shared GROMACS steps from
:class:`sim_prep.base.SimulationPrepper` and adds the apo-specific
MDP template set and config-file update logic.

Config sources
--------------
Accepts configuration as keyword arguments or from a YAML/JSON file
via :func:`sim_prep.config.load_config`.  Both are equivalent::

    # From kwargs
    sim = ApoSimPrepper(protein_name="hsp90", sim_len=100, ...)

    # From YAML (via config loader)
    sim = load_config("apo.yaml")

Typical pipeline
----------------
::

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sim_prep.base import (
    LIMITS,
    VALID_BOX_SHAPES,
    VALID_NEG_IONS,
    VALID_POS_IONS,
    SimulationPrepper,
)


class ApoSimPrepper(SimulationPrepper):
    """
    Preparation workflow for apo (protein-only) simulations.

    Uses AMBER99SB-ILDN force field with TIP3P water by default.
    All MDP templates are read from ``config/gmx/apo/``.

    Parameters
    ----------
    protein_name : str
        Stem of the input PDB file (without ``.pdb`` extension).
    sim_len : float
        Production run length in nanoseconds.
    bx_dim : float
        Box padding in nanometres (distance from protein surface to
        box edge).
    bx_shp : str
        Box shape — one of ``"cubic"``, ``"dodecahedron"``,
        ``"triclinic"``, ``"octahedron"``.
    md_name : str
        Stem used for all production-run output files.
    pos_ion : str, optional
        Positive counter-ion.  Defaults to ``"NA"``.
    neg_ion : str, optional
        Negative counter-ion.  Defaults to ``"CL"``.
    work_dir : str or Path, optional
        Working directory.  Defaults to ``Path.cwd()``.
    gmx_executable : str, optional
        GROMACS binary name or full path.  Defaults to ``"gmx"``.
    index_file : str, optional
        Path to a custom ``.ndx`` file.  Defaults to ``None``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Point to the apo MDP template directory
        self.config_dir: Path = (
            self.script_directory.parent / "md-configs" / "apo"
        )

        # MDP template stems (filenames without .mdp extension)
        self.mdp_files: dict[str, str] = {
            "ions": "ions",
            "em":   "em",
            "nvt":  "nvt",
            "npt":  "npt",
            "md":   "md",
        }

        # AMBER99SB-ILDN (6) + TIP3P (1)
        # Override ff_selection after construction to use a different FF
        self.ff_selection: str = "6\n1\n"
        # Group 13 = SOL in a standard protein-water system
        self.genion_sele: str = "13"

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Validate ``self.config`` for an apo simulation.

        Checks all required fields and their types and ranges.  Collects
        every problem found before raising so the user sees the full list
        of issues in a single error message.

        Raises
        ------
        ValueError
            If any required parameter is missing, the wrong type, or
            outside the permitted range.
        """
        cfg = self.config
        errors: list[str] = []

        # protein_name — required, non-empty string
        if not isinstance(cfg.get("protein_name"), str) or not cfg["protein_name"].strip():
            errors.append("protein_name must be a non-empty string")

        # sim_len — required, numeric, within range
        sim_len = cfg.get("sim_len")
        try:
            sim_len = float(sim_len)  # coerce int or string-number from YAML
        except (TypeError, ValueError):
            errors.append("sim_len must be a number (nanoseconds)")
            sim_len = None
        if sim_len is not None:
            lo, hi = LIMITS["sim_len"]
            if not (lo <= sim_len <= hi):
                errors.append(f"sim_len must be between {lo} and {hi} ns, got {sim_len}")

        # bx_shp — required, one of the valid shapes
        if cfg.get("bx_shp") not in VALID_BOX_SHAPES:
            errors.append(
                f"bx_shp must be one of {VALID_BOX_SHAPES}, "
                f"got {cfg.get('bx_shp')!r}"
            )

        # bx_dim — required, numeric, within range
        bx_dim = cfg.get("bx_dim")
        try:
            bx_dim = float(bx_dim)
        except (TypeError, ValueError):
            errors.append("bx_dim must be a number (nm)")
            bx_dim = None
        if bx_dim is not None:
            lo, hi = LIMITS["bx_dim"]
            if not (lo <= bx_dim <= hi):
                errors.append(f"bx_dim must be between {lo} and {hi} nm, got {bx_dim}")

        # md_name — required, non-empty string
        if not isinstance(cfg.get("md_name"), str) or not cfg["md_name"].strip():
            errors.append("md_name must be a non-empty string")

        # ions — optional with defaults; validate only if explicitly provided
        if cfg.get("pos_ion", "NA") not in VALID_POS_IONS:
            errors.append(
                f"pos_ion must be one of {VALID_POS_IONS}, "
                f"got {cfg.get('pos_ion')!r}"
            )
        if cfg.get("neg_ion", "CL") not in VALID_NEG_IONS:
            errors.append(
                f"neg_ion must be one of {VALID_NEG_IONS}, "
                f"got {cfg.get('neg_ion')!r}"
            )

        if errors:
            raise ValueError(
                "ApoSimPrepper config validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def update_config_files(self) -> None:
        """
        Patch ``md.mdp`` with the correct ``nsteps`` value derived from
        ``self.sim_length``.

        For apo simulations only the production MDP needs updating;
        equilibration MDPs use fixed step counts that are set in the
        template files.
        """
        md_mdp = self.working_dir / f"{self.mdp_files['md']}.mdp"
        self._patch_nsteps(md_mdp)
        print(f"✓  Updated {md_mdp.name}")
