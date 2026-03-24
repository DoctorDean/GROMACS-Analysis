"""
apo.py
======
Simulation preparation for apo (protein-only) systems.

The ``ApoSimPrepper`` class inherits all shared GROMACS steps from
:class:`sim_prep.base.SimulationPrepper` and adds the apo-specific
MDP template set and config-file update logic.

Typical pipeline
----------------
::

    from sim_prep.apo import ApoSimPrepper

    sim = ApoSimPrepper(
        protein_name="hsp90",
        sim_len=100,
        bx_dim=1.0,
        bx_shp="dodecahedron",
        md_name="md_production",
        pos_ion="NA",
        neg_ion="CL",
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

from base import (
    LIMITS,
    VALID_BOX_SHAPES,
    VALID_NEG_IONS,
    VALID_POS_IONS,
    SimulationPrepper,
)


class ApoSimPrepper(SimulationPrepper):
    """
    Preparation workflow for apo protein simulations.

    Uses the AMBER99SB-ILDN force field with TIP3P water by default.
    All MDP templates are read from ``md-configs/apo/``.

    Parameters
    ----------
    protein_name : str
        Stem of the input PDB file (without ``.pdb`` extension).
    sim_len : float
        Production run length in nanoseconds.
    bx_dim : float
        Box padding in nanometres (distance from protein to box edge).
    bx_shp : str
        Box shape — ``"cubic"``, ``"dodecahedron"``, etc.
    md_name : str
        Stem used for all production-run output files.
    pos_ion : str, optional
        Positive counter-ion.  Defaults to ``"NA"``.
    neg_ion : str, optional
        Negative counter-ion.  Defaults to ``"CL"``.
    work_dir : str or Path, optional
        Working directory.  Defaults to ``Path.cwd()``.
    **kwargs
        Passed to :class:`~sim_prep.base.SimulationPrepper`.
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

        # AMBER99SB-ILDN (6) + TIP3P (1); override if using a different FF
        self.ff_selection: str = "6\n1\n"
        # Group 13 = SOL in a standard protein-water system
        self.genion_sele: str = "13"

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Validate the constructor kwargs for an apo simulation.

        Raises
        ------
        ValueError
            If any required parameter is missing or out of range.
        """
        cfg = self.config
        errors: list[str] = []

        # protein_name
        if not isinstance(cfg.get("protein_name"), str) or not cfg["protein_name"]:
            errors.append("protein_name must be a non-empty string")

        # sim_len
        sim_len = cfg.get("sim_len")
        if not isinstance(sim_len, (int, float)):
            errors.append("sim_len must be a number (nanoseconds)")
        elif not (LIMITS["sim_len"][0] <= sim_len <= LIMITS["sim_len"][1]):
            lo, hi = LIMITS["sim_len"]
            errors.append(f"sim_len must be between {lo} and {hi} ns")

        # bx_shp
        if cfg.get("bx_shp") not in VALID_BOX_SHAPES:
            errors.append(f"bx_shp must be one of {VALID_BOX_SHAPES}")

        # bx_dim
        bx_dim = cfg.get("bx_dim")
        if not isinstance(bx_dim, (int, float)):
            errors.append("bx_dim must be a number (nm)")
        elif not (LIMITS["bx_dim"][0] <= bx_dim <= LIMITS["bx_dim"][1]):
            lo, hi = LIMITS["bx_dim"]
            errors.append(f"bx_dim must be between {lo} and {hi} nm")

        # md_name
        if not isinstance(cfg.get("md_name"), str) or not cfg["md_name"]:
            errors.append("md_name must be a non-empty string")

        # ions
        if cfg.get("pos_ion", "NA") not in VALID_POS_IONS:
            errors.append(f"pos_ion must be one of {VALID_POS_IONS}")
        if cfg.get("neg_ion", "CL") not in VALID_NEG_IONS:
            errors.append(f"neg_ion must be one of {VALID_NEG_IONS}")

        if errors:
            raise ValueError(
                "ApoSimPrepper config validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def update_config_files(self) -> None:
        """
        Patch ``md.mdp`` in the working directory with the correct
        ``nsteps`` value derived from ``self.sim_length``.

        For apo simulations only the production MDP needs updating;
        the equilibration MDPs use fixed step counts.
        """
        md_mdp = self.working_dir / f"{self.mdp_files['md']}.mdp"
        self._patch_nsteps(md_mdp)
        print(f"✓  Updated {md_mdp.name}")
