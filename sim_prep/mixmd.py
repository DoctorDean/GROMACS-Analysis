"""
mixmd.py
========
Simulation preparation for mixed-solvent (MixMD) systems.

MixMD simulations co-solvate a protein with small organic probe
molecules (ligands) at defined concentrations.  Each probe is
parameterised individually with AmberTools/ACPYPE, then copies of each
are inserted into the simulation box with ``gmx insert-molecules``.

Typical pipeline
----------------
::

    from sim_prep.mixmd import MixMDPrepper

    sim = MixMDPrepper(
        protein_name="hsp90",
        sim_len=100,
        bx_dim=1.5,
        bx_shp="dodecahedron",
        md_name="md_production",
        pos_ion="NA",
        neg_ion="CL",
        ligands=[
            {"code": "ACT", "number": 50, "smiles": "CC(=O)[O-]"},
            {"code": "MSE", "number": 20, "smiles": "CSCC[C@@H](N)C(=O)O"},
        ],
        work_dir="/data/simulations/hsp90_mixmd",
    )
    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()
    sim.clean_pdb_file()
    sim.protein_pdb2gmx()
    sim.set_new_box()
    sim.param_all_ligands()
    sim.top2itp()
    sim.merge_atomtypes()
    sim.build_mixmd()
    sim.solvate()
    sim.minimise_system()
    sim.nvt_equilibration()
    sim.npt_equilibration()
    sim.production_run()
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from sim_prep.base import (
    LIMITS,
    VALID_BOX_SHAPES,
    VALID_NEG_IONS,
    VALID_POS_IONS,
    SimulationPrepper,
)
from utils.amber_params import AmberParameteriser


class MixMDPrepper(SimulationPrepper):
    """
    Preparation workflow for mixed-solvent MD (MixMD) simulations.

    Parameters
    ----------
    protein_name : str
        Stem of the protein PDB file (without ``.pdb``).
    ligands : list[dict]
        List of probe-molecule definitions.  Each entry must contain:

        ``code``   (str)   Three-letter residue code (e.g. ``"ACT"``).
        ``number`` (int)   Number of copies to insert.
        ``smiles`` (str)   SMILES string for the probe molecule.

    sim_len : float
        Production run length in nanoseconds.
    bx_dim : float
        Box padding in nanometres.
    bx_shp : str
        Box shape.
    md_name : str
        Stem for production-run output files.
    pos_ion : str, optional
        Positive counter-ion.  Defaults to ``"NA"``.
    neg_ion : str, optional
        Negative counter-ion.  Defaults to ``"CL"``.
    work_dir : str or Path, optional
        Working directory.  Defaults to ``Path.cwd()``.
    **kwargs
        Passed to :class:`~sim_prep.base.SimulationPrepper`.

    Raises
    ------
    ValueError
        If ``protein_name`` or ``ligands`` are absent from kwargs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        raw_ligands: list[dict] = self.config.get("ligands") or []
        self.ligands: list[dict] = raw_ligands

        # Build derived lists — use .get() throughout so that malformed
        # ligand dicts don't crash __init__ before validate_config() can
        # collect and report all errors in a structured way
        self.ligand_codes:   list[str]           = [l.get("code",   "") for l in raw_ligands]
        self.ligand_numbers: list[int]            = [l.get("number", 0)  for l in raw_ligands]
        self.smiles_strings: list[Optional[str]]  = [l.get("smiles")     for l in raw_ligands]

        # MDP templates and config directory
        self.config_dir = (
            self.script_directory.parent / "config" / "gmx" / "mixmd"
        )
        self.mdp_files: dict[str, str] = {
            "ions": "ions_mix",
            "em":   "em_mix",
            "nvt":  "nvt_mix",
            "npt":  "npt_mix",
            "md":   "md_mix",
        }

        # NOTE: early validation of required fields is deferred to
        # validate_config() so load_config() surfaces all errors together.

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Validate ``self.config`` for a MixMD simulation.

        Collects every problem before raising so the user sees the full
        list in a single error message.

        Raises
        ------
        ValueError
            If any required parameter is missing, wrong type, or out of range.
        """
        cfg = self.config
        errors: list[str] = []

        # protein_name
        if not isinstance(cfg.get("protein_name"), str) or not cfg["protein_name"].strip():
            errors.append("protein_name must be a non-empty string")

        # sim_len — coerce to float
        sim_len = cfg.get("sim_len")
        try:
            sim_len = float(sim_len)
        except (TypeError, ValueError):
            errors.append("sim_len must be a number (nanoseconds)")
            sim_len = None
        if sim_len is not None:
            lo, hi = LIMITS["sim_len"]
            if not (lo <= sim_len <= hi):
                errors.append(f"sim_len must be between {lo} and {hi} ns, got {sim_len}")

        # bx_shp
        if cfg.get("bx_shp") not in VALID_BOX_SHAPES:
            errors.append(
                f"bx_shp must be one of {VALID_BOX_SHAPES}, "
                f"got {cfg.get('bx_shp')!r}"
            )

        # bx_dim — coerce to float
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

        # ions
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

        # ligands — required non-empty list; smiles is optional per entry
        ligands = cfg.get("ligands") or []
        if not ligands:
            errors.append(
                "ligands must be a non-empty list of probe molecule definitions"
            )
        else:
            for i, lig in enumerate(ligands):
                for required_field in ("code", "number"):
                    if required_field not in lig or lig[required_field] is None:
                        errors.append(
                            f"ligands[{i}] missing required field '{required_field}'"
                        )
                # smiles is optional — a pre-built {code}.pdb may be provided instead
                # No error if smiles is absent or null; param_all_ligands() will
                # raise FileNotFoundError at runtime if neither smiles nor PDB exists

        if errors:
            raise ValueError(
                "MixMDPrepper config validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def update_config_files(self) -> None:
        """
        Patch ``md_mix.mdp`` with the correct ``nsteps`` value.
        """
        self._patch_nsteps(self.working_dir / f"{self.mdp_files['md']}.mdp")
        print(f"✓  Updated {self.mdp_files['md']}.mdp")

    # ------------------------------------------------------------------
    # MixMD-specific pipeline steps
    # ------------------------------------------------------------------

    def param_all_ligands(
        self,
        charge_method: str = "bcc",
        atom_type: str = "gaff",
        net_charges: Optional[dict[str, int]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Parameterise every probe molecule in ``self.ligands`` using
        AmberTools and ACPYPE.

        Each ligand is processed independently through the full
        pipeline (protonation → antechamber → parmchk2 → tleap →
        ACPYPE).  Because ``AmberParameteriser`` uses the ligand code
        directly as the residue name from the start, no post-hoc
        directory or file renaming is required.

        Parameters
        ----------
        charge_method : str, optional
            Partial charge method.  Defaults to ``"bcc"`` (AM1-BCC).
        atom_type : str, optional
            GAFF atom-type scheme.  Defaults to ``"gaff"``.
        net_charges : dict[str, int], optional
            Per-ligand net formal charges, keyed by ligand code.
            Example: ``{"ACT": -1, "MSE": 0}``.  Ligands not listed
            use automatic charge inference.
        verbose : bool, optional
            Print subprocess stdout in real time.  Defaults to ``False``.

        Output files (per ligand)
        -------------------------
        ``{code}.amb2gmx/{code}_GMX.top``,
        ``{code}.amb2gmx/{code}_GMX.gro``,
        ``{code}.amb2gmx/posre_{code}.itp``,
        plus copies of all the above in ``working_dir``.
        """
        net_charges = net_charges or {}

        for code in self.ligand_codes:
            parameteriser = AmberParameteriser(
                ligand_code=code,
                working_dir=self.working_dir,
                charge_method=charge_method,
                atom_type=atom_type,
                net_charge=net_charges.get(code),
                verbose=verbose,
            )
            parameteriser.run()

            # Copy outputs to the working directory for easy access
            output_dir = self.working_dir / f"{code}.amb2gmx"
            for item in output_dir.iterdir():
                shutil.copy(item, self.working_dir)
                print(f"✓  Copied {item.name} → working directory")

    def top2itp(self) -> None:
        """
        Convert each ligand's GROMACS ``.top`` file into an include
        topology (``.itp``) by extracting the section from
        ``[ moleculetype ]`` up to (but not including) ``[ system ]``.

        Output files
        ------------
        ``{code}_GMX.itp``  (one per ligand, in ``working_dir``)
        """
        for code in self.ligand_codes:
            top_path = self.working_dir / f"{code}_GMX.top"
            itp_path = self.working_dir / f"{code}_GMX.itp"

            lines = top_path.read_text().splitlines(keepends=True)

            mol_start = sys_start = None
            for i, line in enumerate(lines):
                if "[ moleculetype ]" in line and mol_start is None:
                    mol_start = i
                if "[ system ]" in line and sys_start is None:
                    sys_start = i
                    break

            if mol_start is None:
                raise ValueError(f"[ moleculetype ] not found in {top_path}")
            if sys_start is None:
                sys_start = len(lines)

            with itp_path.open("w") as fh:
                fh.write(lines[0])                           # title comment
                fh.writelines(lines[mol_start:sys_start])

            print(f"✓  ITP written → {itp_path.name}")

    def merge_atomtypes(self) -> None:
        """
        Collect ``[ atomtypes ]`` entries from every ligand ``.top``
        file, deduplicate them, and write them into the first ligand's
        ITP file together with a ``[ defaults ]`` header.

        This ensures that the combined topology has exactly one
        ``[ atomtypes ]`` section with no duplicate entries even when
        multiple probe molecules share common atom types.

        Output files
        ------------
        ``{ligand_codes[0]}_GMX.itp`` (modified in-place)
        """
        all_atomtypes: set[str] = set()

        for code in self.ligand_codes:
            top_path = self.working_dir / f"{code}_GMX.top"
            lines = top_path.read_text().splitlines(keepends=True)

            inside = False
            for line in lines:
                if "[ atomtypes ]" in line:
                    inside = True
                elif inside and line.strip() == "":
                    inside = False
                elif inside:
                    all_atomtypes.add(line.strip())

        first_itp = self.working_dir / f"{self.ligand_codes[0]}_GMX.itp"
        original_lines = first_itp.read_text().splitlines(keepends=True)

        merged: list[str] = [
            original_lines[0],
            "\n[ defaults ]\n",
            "; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ\n",
            "1               2               yes             0.5     0.8333333333\n\n",
            "[ atomtypes ]\n",
        ]
        merged.extend(sorted(f"{entry}\n" for entry in all_atomtypes))
        merged.append("\n")
        merged.extend(original_lines[1:])

        first_itp.write_text("".join(merged))
        print(f"✓  Non-redundant [ atomtypes ] merged into {first_itp.name}")

    def build_mixmd(self) -> None:
        """
        Sequentially insert each probe molecule into the simulation box
        using ``gmx insert-molecules``, then rename the final file to
        ``mixmd.gro`` and update the topology molecule counts.

        Output files
        ------------
        ``mixmd.gro``   Final MixMD system prior to solvation.
        ``topol.top``   Updated with molecule counts for each probe.
        """
        current = self.working_dir / "newbox.gro"

        for code, number in zip(self.ligand_codes, self.ligand_numbers):
            out_name = str(current).replace("newbox", f"newbox_{code}")
            output = Path(out_name)

            self._run(
                [
                    self.gmx, "insert-molecules",
                    "-f", str(current),
                    "-ci", f"{code}_GMX.gro",
                    "-o", str(output),
                    "-nmol", str(number),
                    "-try", "1000",
                ],
                label=f"insert-molecules ({number}× {code})",
            )
            self.update_topology_molecules(code, number)
            current = output

        final = self.working_dir / "mixmd.gro"
        current.rename(final)
        print(f"✓  Final MixMD system → {final.name}")

    def solvate(self) -> None:
        """
        Solvate the MixMD system and add counter-ions.

        Reads from ``mixmd.gro`` rather than the standard ``newbox.gro``
        used by the base class.

        Output files
        ------------
        ``solv.gro``, ``ions.tpr``, ``solv_ions.gro``
        """
        ions_mdp = f"{self.mdp_files['ions']}.mdp"

        self._run(
            [self.gmx, "solvate",
             "-cp", "mixmd.gro",
             "-cs", "spc216.gro",
             "-p", "topol.top",
             "-o", "solv.gro"],
            label="solvate",
        )
        self._run(
            [self.gmx, "grompp",
             "-f", ions_mdp,
             "-c", "solv.gro",
             "-p", "topol.top",
             "-o", "ions.tpr"],
            label="grompp (ions)",
        )
        self._run(
            [self.gmx, "genion",
             "-s", "ions.tpr",
             "-o", "solv_ions.gro",
             "-p", "topol.top",
             "-pname", self.pos_ion,
             "-nname", self.neg_ion,
             "-neutral"],
            label="genion",
            stdin="15",   # SOL group in a protein + multi-ligand system
        )
