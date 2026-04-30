"""
complex.py
==========
Simulation preparation for protein–ligand complex systems.

The ``CplxSimPrepper`` class inherits all shared GROMACS steps from
:class:`sim_prep.base.SimulationPrepper` and adds the ligand-specific
steps required to parameterise a small molecule with AmberTools/ACPYPE
and assemble the protein–ligand complex topology.

Three input paths are supported
--------------------------------

**Path A — AutoDock complex** (docked PDB with ligand inside):
::

    sim.process_autodocked_complex()   # splits protein + ligand, runs pdb2gmx
    sim.param_with_amber()
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()

**Path B — Native structure** (PDB, mmCIF, or GRO from RCSB / AlphaFold / Glide):
::

    sim.prepare_from_structure(
        protein_input="6lu7.cif",          # .pdb / .cif / .mmcif / .gro
        ligand_input="N3.sdf",             # .pdb / .mol2 / .sdf / .mol
                                           # or "smiles:CC(=O)..."
        keep_residues=["ZN"],              # metal ions to retain
    )
    sim.param_with_amber()
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()

**Path C — Pre-separated files** (protein and ligand already in separate files):
::

    sim.prepare_from_separate_files(
        protein_file="protein_clean.pdb",
        ligand_file="ligand.mol2",         # .pdb / .mol2 / .sdf / .mol
    )
    sim.param_with_amber()
    sim.gro2itp()
    sim.generate_ligand_ndx()
    sim.build_gmx_complex()
    sim.build_complex_topology()

All three paths converge at ``param_with_amber()`` and share the
remaining solvation, equilibration, and production-run steps.

Full example (AutoDock path)
-----------------------------
::

    from sim_prep.complex import CplxSimPrepper

    sim = CplxSimPrepper(
        protein_name="hsp90",
        ligand_code="ATP",
        param_ligand_name="UNL",
        remove_ligands=["UNL"],
        sim_len=100,
        bx_dim=1.0,
        bx_shp="dodecahedron",
        md_name="md_production",
        pos_ion="NA",
        neg_ion="CL",
        work_dir="/data/simulations/hsp90_ATP",
    )
    sim.validate_config()
    sim.assign_attributes()
    sim.copy_config_files()
    sim.update_config_files()
    sim.process_autodocked_complex()
    sim.param_with_amber()
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
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from shutil import copy2
from typing import Any, Optional

import subprocess

from sim_prep.base import (
    LIMITS,
    VALID_BOX_SHAPES,
    VALID_NEG_IONS,
    VALID_POS_IONS,
    SimulationPrepper,
)
from utils.amber_params import AmberParameteriser
from utils.structure_io import prepare_structure


class CplxSimPrepper(SimulationPrepper):
    """
    Preparation workflow for protein–ligand complex simulations.

    Supports three input paths — see module docstring for full details:

    - **Path A**: AutoDock complex PDB → :meth:`process_autodocked_complex`
    - **Path B**: Native PDB / mmCIF / GRO + separate ligand file or SMILES
                  → :meth:`prepare_from_structure`
    - **Path C**: Pre-separated protein and ligand files
                  → :meth:`prepare_from_separate_files`

    All paths converge at :meth:`param_with_amber` and share all
    subsequent solvation, equilibration, and production steps.

    Parameters
    ----------
    protein_name : str
        Stem used for output files (without extension).  For Path A,
        this must also match the input complex PDB filename stem.
    ligand_code : str
        Three-letter residue code for the ligand (e.g. ``"ATP"``).
    param_ligand_name : str, optional
        Residue name used by the docking software (AutoDock Vina uses
        ``"UNL"`` by default).  Only relevant for Path A.
    remove_ligands : list[str], optional
        Residue names to strip when preparing the protein-only PDB for
        ``pdb2gmx``.  Defaults to ``["UNL"]``.  Only relevant for
        Path A.
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
        If ``protein_name`` or ``ligand_code`` are absent from kwargs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.ligand_code: Optional[str] = self.config.get("ligand_code")
        self.param_ligand_name: str     = self.config.get("param_ligand_name", "UNL")
        self.remove_ligands: list[str]  = self.config.get("remove_ligands", ["UNL"])

        # MDP templates and config directory
        self.config_dir = (
            self.script_directory.parent / "md-configs" / "complex"
        )

        self.mdp_files: dict[str, str] = {
            "ions": "ions_prot_lig",
            "em":   "em_prot_lig",
            "nvt":  "nvt_prot_lig",
            "npt":  "npt_prot_lig",
            "md":   "md_prot_lig",
        }

        # Path to the parameterisation shell script
        self.param_with_amber_script: Path = (
            self.script_directory.parent / "utils" / "param_with_amber.sh"
        )

        # Ligand file handles (populated after param_with_amber / gro2itp)
        self.ligand_dir:       Optional[Path] = None
        self.ligand_top_file:  Optional[Path] = None
        self.ligand_itp_file:  Optional[Path] = None
        self.ligand_gro_file:  Optional[Path] = None
        self.ligand_posre_file: Optional[Path] = None

        # Group 15 = SOL in a protein-ligand-water system
        self.genion_sele = "15"

        if not self.config.get("protein_name") or not self.ligand_code:
            raise ValueError(
                "CplxSimPrepper requires both 'protein_name' and 'ligand_code' in kwargs."
            )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Validate constructor kwargs for a protein–ligand complex simulation.

        Raises
        ------
        ValueError
            If any required parameter is missing or out of range.
        """
        cfg = self.config
        errors: list[str] = []

        for key in ("protein_name", "ligand_code", "md_name"):
            if not isinstance(cfg.get(key), str) or not cfg[key]:
                errors.append(f"{key} must be a non-empty string")

        sim_len = cfg.get("sim_len")
        if not isinstance(sim_len, (int, float)):
            errors.append("sim_len must be a number (nanoseconds)")
        elif not (LIMITS["sim_len"][0] <= sim_len <= LIMITS["sim_len"][1]):
            lo, hi = LIMITS["sim_len"]
            errors.append(f"sim_len must be between {lo} and {hi} ns")

        if cfg.get("bx_shp") not in VALID_BOX_SHAPES:
            errors.append(f"bx_shp must be one of {VALID_BOX_SHAPES}")

        bx_dim = cfg.get("bx_dim")
        if not isinstance(bx_dim, (int, float)):
            errors.append("bx_dim must be a number (nm)")
        elif not (LIMITS["bx_dim"][0] <= bx_dim <= LIMITS["bx_dim"][1]):
            lo, hi = LIMITS["bx_dim"]
            errors.append(f"bx_dim must be between {lo} and {hi} nm")

        if cfg.get("pos_ion", "NA") not in VALID_POS_IONS:
            errors.append(f"pos_ion must be one of {VALID_POS_IONS}")
        if cfg.get("neg_ion", "CL") not in VALID_NEG_IONS:
            errors.append(f"neg_ion must be one of {VALID_NEG_IONS}")

        if errors:
            raise ValueError(
                "CplxSimPrepper config validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def update_config_files(self) -> None:
        """
        Patch complex MDP files with the correct ``tc-grps`` coupling
        groups (``Protein_{ligand_code}`` + ``Water_and_ions``) and the
        correct ``nsteps`` for the production run.
        """
        tc_line = (
            f"tc-grps                 = Protein_{self.ligand_code}"
            f" Water_and_ions    ; two coupling groups\n"
        )
        equilibration_mdps = [
            f"{self.mdp_files['nvt']}.mdp",
            f"{self.mdp_files['npt']}.mdp",
            f"{self.mdp_files['md']}.mdp",
        ]

        for mdp_name in equilibration_mdps:
            mdp_path = self.working_dir / mdp_name
            lines = mdp_path.read_text().splitlines(keepends=True)
            patched: list[str] = []
            for line in lines:
                if line.startswith("tc-grps"):
                    line = tc_line
                patched.append(line)
            mdp_path.write_text("".join(patched))
            print(f"✓  tc-grps updated in {mdp_name}")

        # Patch nsteps only in the production MDP
        self._patch_nsteps(self.working_dir / f"{self.mdp_files['md']}.mdp")

    # ------------------------------------------------------------------
    # Complex-specific pipeline steps
    # ------------------------------------------------------------------

    def process_autodocked_complex(self) -> None:
        """
        Prepare the AutoDock output PDB for GROMACS parameterisation.

        Steps
        -----
        1. Write ``{protein_name}_clean.pdb`` with all residues listed in
           ``remove_ligands`` stripped out (protein-only, for pdb2gmx).
        2. Write ``{ligand_code}.pdb`` containing only the ligand
           residue, renamed from ``param_ligand_name`` to ``ligand_code``.
        3. Call :meth:`protein_pdb2gmx` on the cleaned protein PDB.

        Output files
        ------------
        ``{protein_name}_clean.pdb``, ``{ligand_code}.pdb``,
        ``{protein_name}_processed.gro``, ``topol.top``, ``posre.itp``
        """
        src_pdb     = self.working_dir / f"{self.protein_name}.pdb"
        clean_pdb   = self.working_dir / f"{self.protein_name}_clean.pdb"
        ligand_pdb  = self.working_dir / f"{self.ligand_code}.pdb"

        with src_pdb.open() as fh:
            all_lines = fh.readlines()

        # Protein: drop all residues in remove_ligands
        with clean_pdb.open("w") as fh:
            fh.writelines(
                l for l in all_lines
                if l[17:20].strip() not in self.remove_ligands
            )

        # Ligand: keep only the target residue, rename to ligand_code
        with ligand_pdb.open("w") as fh:
            fh.writelines(
                l.replace(self.param_ligand_name, self.ligand_code)
                for l in all_lines
                if self.param_ligand_name in l
            )

        print(f"✓  Cleaned PDB → {clean_pdb.name}")
        print(f"✓  Ligand PDB  → {ligand_pdb.name}")
        self.protein_pdb2gmx()

    # ------------------------------------------------------------------
    # Path B — native structure input (PDB, mmCIF, GRO + ligand file)
    # ------------------------------------------------------------------

    def prepare_from_structure(
        self,
        protein_input: str,
        ligand_input: str,
        keep_residues: Optional[list[str]] = None,
        remove_hetatm: bool = True,
        remove_waters: bool = True,
    ) -> None:
        """
        Prepare a complex from a native protein structure and a separate
        ligand file (or SMILES string).

        Use this path when starting from an RCSB PDB / mmCIF download,
        an AlphaFold model, or a Glide/GOLD output where the protein and
        ligand files are in separate formats.

        Steps
        -----
        1. Convert and clean the protein structure to
           ``{protein_name}_clean.pdb`` using :func:`~utils.structure_io.prepare_structure`.
        2. Convert the ligand to ``{ligand_code}.pdb``.
        3. Run ``gmx pdb2gmx`` on the protein PDB.

        Parameters
        ----------
        protein_input : str
            Path to the protein structure file.  Accepted formats:
            ``.pdb``, ``.cif`` / ``.mmcif``, ``.gro``.
        ligand_input : str
            Path to the ligand file, or an inline SMILES string prefixed
            with ``"smiles:"``.  Accepted file formats: ``.pdb``,
            ``.mol2``, ``.sdf``, ``.mol``.
        keep_residues : list[str], optional
            Residue names to retain even when ``remove_hetatm=True``.
            Use this to keep metal ions co-crystallised with the protein
            (e.g. ``["ZN", "MG", "FE"]``).  These residues will need
            manual parameter files or custom force-field entries.
        remove_hetatm : bool, optional
            Strip ``HETATM`` records from the protein structure (other
            than those in ``keep_residues``).  Defaults to ``True``.
        remove_waters : bool, optional
            Strip water molecules from the protein structure.  Defaults
            to ``True``.  Crystal waters are generally removed before MD
            solvation; retain them only if they are known to be
            functionally important.

        Output files
        ------------
        ``{protein_name}_clean.pdb``, ``{ligand_code}.pdb``,
        ``{protein_name}_processed.gro``, ``topol.top``, ``posre.itp``

        Notes
        -----
        After this method returns, call :meth:`param_with_amber` to
        parameterise the ligand, then continue with :meth:`gro2itp`,
        :meth:`generate_ligand_ndx`, :meth:`build_gmx_complex`, and
        :meth:`build_complex_topology`.
        """
        keep_residues = keep_residues or []

        # Convert protein to clean PDB
        prepare_structure(
            protein_input,
            output_name=f"{self.protein_name}_clean",
            work_dir=self.working_dir,
            remove_hetatm=remove_hetatm,
            remove_waters=remove_waters,
            keep_residues=keep_residues,
            gmx_executable=self.gmx,
        )

        # Convert ligand to PDB named {ligand_code}.pdb
        prepare_structure(
            ligand_input,
            output_name=self.ligand_code,
            work_dir=self.working_dir,
            ligand_code=self.ligand_code,
            gmx_executable=self.gmx,
        )

        print(f"✓  Protein → {self.protein_name}_clean.pdb")
        print(f"✓  Ligand  → {self.ligand_code}.pdb")
        self.protein_pdb2gmx()

    # ------------------------------------------------------------------
    # Path C — pre-separated protein and ligand files
    # ------------------------------------------------------------------

    def prepare_from_separate_files(
        self,
        protein_file: str,
        ligand_file: str,
        keep_residues: Optional[list[str]] = None,
        remove_hetatm: bool = True,
        remove_waters: bool = True,
    ) -> None:
        """
        Prepare a complex from protein and ligand files that are already
        separated (e.g. Schrödinger Glide output, manual preparation,
        or any workflow that delivers two distinct structure files).

        This is a thin convenience wrapper around
        :meth:`prepare_from_structure` — it accepts the same argument
        types and produces identical output files.  The distinction
        exists to make intent explicit in pipeline scripts.

        Parameters
        ----------
        protein_file : str
            Path to the protein-only structure file.  Accepted formats:
            ``.pdb``, ``.cif`` / ``.mmcif``, ``.gro``.
        ligand_file : str
            Path to the ligand-only file.  Accepted formats: ``.pdb``,
            ``.mol2``, ``.sdf``, ``.mol``.  SMILES strings (``"smiles:..."``)
            are also accepted.
        keep_residues : list[str], optional
            Residue names in the protein file to retain alongside
            ``ATOM`` records (e.g. metal ions: ``["ZN", "MG"]``).
        remove_hetatm : bool, optional
            Strip ``HETATM`` from the protein file.  Defaults to
            ``True``.
        remove_waters : bool, optional
            Strip waters from the protein file.  Defaults to ``True``.

        Output files
        ------------
        ``{protein_name}_clean.pdb``, ``{ligand_code}.pdb``,
        ``{protein_name}_processed.gro``, ``topol.top``, ``posre.itp``
        """
        self.prepare_from_structure(
            protein_input=protein_file,
            ligand_input=ligand_file,
            keep_residues=keep_residues,
            remove_hetatm=remove_hetatm,
            remove_waters=remove_waters,
        )

    def param_with_amber(
        self,
        charge_method: str = "bcc",
        atom_type: str = "gaff",
        net_charge: Optional[int] = None,
        multiplicity: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Parameterise the ligand with AmberTools and ACPYPE.

        Runs the full parameterisation pipeline natively in Python:

        1. **Protonate** — Open Babel adds missing hydrogens.
        2. **Clean**     — CONECT records are stripped from the PDB.
        3. **antechamber** — AM1-BCC partial charges + MOL2 file.
        4. **SQM check** — verifies the semi-empirical QM job converged.
        5. **parmchk2**  — checks for missing GAFF parameters.
        6. **tleap**     — builds Amber PRMTOP/RST7 topology.
        7. **ACPYPE**    — converts Amber topology to GROMACS format.

        Parameters
        ----------
        charge_method : str, optional
            Partial charge method for ``antechamber``.  Defaults to
            ``"bcc"`` (AM1-BCC).  Other options: ``"gas"`` (Gasteiger),
            ``"mul"`` (Mulliken from a QM run).
        atom_type : str, optional
            GAFF atom-type scheme.  Defaults to ``"gaff"``.  Use
            ``"gaff2"`` for the updated parameterisation.
        net_charge : int, optional
            Net formal charge of the ligand.  If ``None``, inferred
            automatically.  Set explicitly for charged ligands.
        multiplicity : int, optional
            Spin multiplicity.  Defaults to ``1`` (singlet).
        verbose : bool, optional
            Print subprocess stdout in real time.  Defaults to ``False``.

        Output files
        ------------
        ``{ligand_code}.amb2gmx/``
            Directory containing ``{ligand_code}_GMX.top``,
            ``{ligand_code}_GMX.gro``, and ``posre_{ligand_code}.itp``.

        Raises
        ------
        RuntimeError
            If a required executable is missing or any step fails.
        """
        parameteriser = AmberParameteriser(
            ligand_code=self.ligand_code,
            working_dir=self.working_dir,
            charge_method=charge_method,
            atom_type=atom_type,
            net_charge=net_charge,
            multiplicity=multiplicity,
            verbose=verbose,
        )
        parameteriser.run()

        lig_dir = self.working_dir / f"{self.ligand_code}.amb2gmx"
        self.ligand_dir        = lig_dir
        self.ligand_top_file   = lig_dir / f"{self.ligand_code}_GMX.top"
        self.ligand_gro_file   = lig_dir / f"{self.ligand_code}_GMX.gro"
        self.ligand_posre_file = lig_dir / f"posre_{self.ligand_code}.itp"

    def build_gmx_complex(self) -> None:
        """
        Merge the processed protein ``.gro`` with the ligand ``.gro``
        into a single ``complex.gro`` file, updating the atom count in
        the header.

        Output files
        ------------
        ``complex.gro``
        """
        protein_gro = self.working_dir / f"{self.protein_name}_processed.gro"
        complex_gro = self.working_dir / "complex.gro"

        with self.ligand_gro_file.open() as fh:
            lig_lines = fh.readlines()
        with protein_gro.open() as fh:
            prot_lines = fh.readlines()

        n_lig  = int(lig_lines[1].strip())
        n_prot = int(prot_lines[1].strip())
        total  = n_lig + n_prot

        print(f"  Protein atoms: {n_prot}  |  Ligand atoms: {n_lig}  |  Total: {total}")

        merged = prot_lines[:-1] + lig_lines[2:-1] + [prot_lines[-1]]
        merged[1] = f"{total}\n"

        with complex_gro.open("w") as fh:
            fh.writelines(merged)

        print(f"✓  complex.gro written ({total} atoms)")

    def gro2itp(self) -> None:
        """
        Extract the ``[ atomtypes ]`` through pre-``[ system ]`` section
        from the ligand ``.top`` file produced by ACPYPE and write it as
        an ITP include file.

        Also copies ``{ligand_code}_GMX.itp``, ``posre_{ligand_code}.itp``,
        and ``{ligand_code}_GMX.gro`` into the working directory.

        Output files
        ------------
        ``{ligand_code}_GMX.itp`` (in ``ligand_dir`` and ``working_dir``)
        """
        new_itp = self.ligand_dir / f"{self.ligand_code}_GMX.itp"

        with self.ligand_top_file.open() as fh:
            lines = fh.readlines()

        atomtypes_start = system_start = None
        for i, line in enumerate(lines):
            if re.match(r"\[ atomtypes \]", line) and atomtypes_start is None:
                atomtypes_start = i
            if re.match(r"\[ system \]", line) and system_start is None:
                system_start = i - 1
                break

        if atomtypes_start is None:
            raise ValueError(f"[ atomtypes ] section not found in {self.ligand_top_file}")
        if system_start is None:
            system_start = len(lines) - 1

        # Write only the [ atomtypes ] → [ system ] slice
        with new_itp.open("w") as fh:
            for line in lines[atomtypes_start : system_start + 1]:
                fh.write(line)

        self.ligand_itp_file = new_itp
        print(f"✓  ITP written → {new_itp}")

        # Copy supporting files to working directory
        for src in (self.ligand_itp_file, self.ligand_posre_file, self.ligand_gro_file):
            try:
                shutil.copy(src, self.working_dir)
                print(f"✓  Copied {src.name} → working directory")
            except FileNotFoundError:
                print(f"✗  {src.name} not found — run param_with_amber() first")

    def generate_ligand_ndx(self) -> None:
        """
        Create a ligand-only index file (heavy atoms only) using
        ``gmx make_ndx``.

        Output files
        ------------
        ``index_{ligand_code}.ndx``
        """
        self._run(
            [
                self.gmx, "make_ndx",
                "-f", f"{self.ligand_code}_GMX.gro",
                "-o", f"index_{self.ligand_code}.ndx",
            ],
            label=f"make_ndx (ligand {self.ligand_code})",
            stdin="0 & ! a H*\nq\n",
        )

    def build_complex_topology(self) -> None:
        """
        Update ``topol.top`` to include ligand force-field parameters,
        position-restraint directives, and the ligand molecule entry.

        A backup of the original topology is written to
        ``backup_topol.top`` before any modifications.

        Output files
        ------------
        ``topol.top`` (modified in-place), ``backup_topol.top``
        """
        backup = self.working_dir / "backup_topol.top"
        if not backup.exists():
            copy2(self.topology, backup)
            print(f"✓  Topology backup → {backup.name}")

        lines = self.topology.read_text().splitlines(keepends=True)

        lig_ff_line    = f'\n; Include ligand parameters\n#include "{self.ligand_code}_GMX.itp"\n'
        lig_posres_line = (
            f'\n; Ligand position restraints\n'
            f'#ifdef POSRES\n'
            f'#include "posre_{self.ligand_code}.itp"\n'
            f'#endif\n'
        )
        lig_mol_line = f"{self.ligand_code}              1\n"

        updated: list[str] = []
        ff_inserted = False
        in_posres   = False

        for line in lines:
            updated.append(line)

            if '#include "amber99sb' in line and not ff_inserted:
                updated.append(lig_ff_line)
                ff_inserted = True

            if "Include Position" in line:
                in_posres = True

            if in_posres and "#endif" in line:
                updated.append(lig_posres_line)
                in_posres = False

        # Append molecule entry before any trailing blank line
        if updated and updated[-1].strip() == "":
            updated.insert(-1, lig_mol_line)
        else:
            updated.append(lig_mol_line)

        self.topology.write_text("".join(updated))
        print(f"✓  Topology updated with {self.ligand_code} parameters")

    def solvate(self) -> None:
        """
        Build the simulation box, solvate, and add counter-ions.

        Steps
        -----
        1. ``gmx editconf`` — centre the complex and set the box.
        2. ``gmx solvate`` — fill with SPC/E water.
        3. ``gmx grompp`` — pre-process for ion addition.
        4. ``gmx genion`` — replace water molecules with counter-ions.

        Output files
        ------------
        ``newbox.gro``, ``solv.gro``, ``ions.tpr``, ``solv_ions.gro``
        """
        self._run(
            [self.gmx, "editconf",
             "-f", "complex.gro",
             "-o", "newbox.gro",
             "-bt", self.box_type,
             "-d", str(self.box_dim)],
            label="editconf (complex box)",
        )
        self._run(
            [self.gmx, "solvate",
             "-cp", "newbox.gro",
             "-cs", "spc216.gro",
             "-p", "topol.top",
             "-o", "solv.gro"],
            label="solvate",
        )
        self._run(
            [self.gmx, "grompp",
             "-f", "ions_prot_lig.mdp",
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
            stdin=self.genion_sele,
        )

    def minimise_system(self, maxwarn: Optional[int] = None) -> None:
        """
        Energy-minimise the complex system.

        Parameters
        ----------
        maxwarn : int, optional
            Passed as ``-maxwarn`` to ``grompp`` when set.
        """
        em_mdp = f"{self.mdp_files['em']}.mdp"
        grompp_cmd = [
            self.gmx, "grompp",
            "-f", em_mdp,
            "-c", "solv_ions.gro",
            "-p", "topol.top",
            "-o", "em.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        self._run(grompp_cmd, label="grompp (EM)")
        self._run([self.gmx, "mdrun", "-v", "-deffnm", "em"], label="mdrun (EM)")
        self.em_complete = True

    def make_prot_lig_ndx(self) -> None:
        """
        Generate a coupled Protein–Ligand index group for use by the
        NVT/NPT thermostat and barostat.

        The GROMACS group ``Protein_<ligand_code>`` is created by merging
        groups 1 (Protein) and 13 (the ligand residue in the default index).

        Output files
        ------------
        ``index.ndx``
        """
        self._run(
            [self.gmx, "make_ndx",
             "-f", "em.gro",
             "-o", "index.ndx"],
            label="make_ndx (Protein–Ligand)",
            stdin=f"1 | 13\nq\n",
        )

    def nvt_equilibration(self, maxwarn: Optional[int] = None) -> None:
        """NVT equilibration using the protein–ligand index file."""
        nvt_mdp = f"{self.mdp_files['nvt']}.mdp"
        grompp_cmd = [
            self.gmx, "grompp",
            "-f", nvt_mdp,
            "-c", "em.gro",
            "-r", "em.gro",
            "-p", "topol.top",
            "-n", "index.ndx",
            "-o", "nvt.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        self._run(grompp_cmd, label="grompp (NVT)")
        self._run([self.gmx, "mdrun", "-v", "-deffnm", "nvt"], label="mdrun (NVT)")
        self.nvt_complete = True

    def npt_equilibration(self, maxwarn: Optional[int] = 1) -> None:
        """NPT equilibration using the protein–ligand index file."""
        npt_mdp = f"{self.mdp_files['npt']}.mdp"
        grompp_cmd = [
            self.gmx, "grompp",
            "-f", npt_mdp,
            "-c", "nvt.gro",
            "-t", "nvt.cpt",
            "-r", "nvt.gro",
            "-p", "topol.top",
            "-n", "index.ndx",
            "-o", "npt.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        self._run(grompp_cmd, label="grompp (NPT)")
        self._run([self.gmx, "mdrun", "-v", "-deffnm", "npt"], label="mdrun (NPT)")
        self.npt_complete = True

    def production_run(
        self,
        maxwarn: Optional[int] = None,
        extra_mdrun_args: Optional[list[str]] = None,
    ) -> None:
        """Production run using the protein–ligand index file."""
        md_mdp = f"{self.mdp_files['md']}.mdp"
        grompp_cmd = [
            self.gmx, "grompp",
            "-f", md_mdp,
            "-c", "npt.gro",
            "-t", "npt.cpt",
            "-p", "topol.top",
            "-n", "index.ndx",
            "-o", f"{self.md_name}.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        mdrun_cmd = [
            self.gmx, "mdrun",
            "-s", f"{self.md_name}.tpr",
            "-v",
            "-deffnm", self.md_name,
            "-update", "cpu",
        ]
        if extra_mdrun_args:
            mdrun_cmd += extra_mdrun_args

        self._run(grompp_cmd, label="grompp (production)")
        self._run(mdrun_cmd,  label=f"mdrun (production – {self.sim_length} ns)")
