"""
utils/amber_params.py
=====================
Native Python replacement for ``param_with_amber.sh``.

Wraps the AmberTools / ACPYPE ligand parameterisation pipeline in a
single ``AmberParameteriser`` class so that the whole workflow — from
protonation to GROMACS-compatible topology — can be driven from Python
without shelling out to a separate Bash script.

Pipeline
--------
The five steps mirror the original shell script exactly:

    1. **Protonate**  – Open Babel adds missing hydrogen atoms.
    2. **Clean**      – ``sed``-equivalent: strip CONECT records that
                        confuse ``antechamber``.
    3. **antechamber** – Generate AM1-BCC partial charges and a MOL2 file.
    4. **parmchk2**   – Check for missing GAFF parameters; write a FRCMOD
                        supplement file.
    5. **tleap**      – Build Amber PRMTOP/RST7 topology files.
    6. **ACPYPE**     – Convert Amber topology to GROMACS format.

All intermediate and output files are written to ``working_dir``.

Dependencies
------------
See ``LIGAND_PARAMETERISATION.md`` in the repository root for full
installation instructions.  In brief:

    conda create -n ambertools python=3.10
    conda activate ambertools
    conda install -c conda-forge ambertools acpype openbabel

Required executables (must be on PATH when Python is invoked):
    - ``obabel``      (Open Babel ≥ 3.1)
    - ``antechamber`` (AmberTools ≥ 23)
    - ``parmchk2``    (AmberTools ≥ 23)
    - ``tleap``       (AmberTools ≥ 23)
    - ``acpype``      (≥ 2022.7)

Example usage
-------------
    from utils.amber_params import AmberParameteriser

    p = AmberParameteriser(
        ligand_code="ATP",
        working_dir="/data/simulations/hsp90_ATP",
    )
    p.run()
    # Output: ATP.amb2gmx/ATP_GMX.top, ATP.amb2gmx/ATP_GMX.gro, …
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Executable availability check
# ---------------------------------------------------------------------------

_REQUIRED_EXECUTABLES = ("obabel", "antechamber", "parmchk2", "tleap", "acpype")


def check_dependencies() -> list[str]:
    """
    Return a list of required executables that are not found on PATH.

    Call this once at startup to give the user an early, clear error
    rather than a confusing ``FileNotFoundError`` mid-pipeline.

    Returns
    -------
    list[str]
        Names of missing executables.  Empty list means all are present.
    """
    import shutil
    return [exe for exe in _REQUIRED_EXECUTABLES if shutil.which(exe) is None]


# ---------------------------------------------------------------------------
# AmberParameteriser
# ---------------------------------------------------------------------------

class AmberParameteriser:
    """
    Parameterise a small-molecule ligand with AmberTools and ACPYPE,
    producing GROMACS-compatible topology files.

    Parameters
    ----------
    ligand_code : str
        Three-letter residue code for the ligand (e.g. ``"ATP"``).
        A PDB file named ``{ligand_code}.pdb`` must exist in
        ``working_dir``.
    working_dir : str or Path, optional
        Directory containing the input PDB and where all outputs are
        written.  Defaults to the current working directory.
    charge_method : str, optional
        Charge method passed to ``antechamber -c``.  Defaults to
        ``"bcc"`` (AM1-BCC, recommended for drug-like molecules).
        Other options: ``"gas"`` (Gasteiger), ``"mul"`` (Mulliken from
        a QM calculation).
    atom_type : str, optional
        Atom type scheme passed to ``antechamber -at``.  Defaults to
        ``"gaff"`` (General Amber Force Field).  Use ``"gaff2"`` for
        the updated parameterisation.
    net_charge : int, optional
        Net formal charge of the ligand.  If ``None`` (default),
        ``antechamber`` infers the charge automatically.  Set this
        explicitly for charged ligands (e.g. ``-1`` for acetate,
        ``+1`` for protonated amines) to avoid incorrect BCC charges.
    multiplicity : int, optional
        Spin multiplicity.  Defaults to ``1`` (singlet, i.e. closed
        shell).  Rarely needs changing for typical drug-like molecules.
    force_field_source : str, optional
        ``leaprc`` source line used in the ``tleap`` input.  Defaults
        to ``"oldff/leaprc.ff99SB"``.  Change to e.g.
        ``"leaprc.protein.ff14SB"`` if you want a newer protein FF
        for the tleap check step (does not affect GROMACS output).
    verbose : bool, optional
        If ``True``, subprocess stdout is printed in real time.
        Defaults to ``False`` (only errors are shown).

    Raises
    ------
    FileNotFoundError
        If the input PDB does not exist.
    RuntimeError
        If any step of the pipeline fails or if required executables
        are missing.
    """

    def __init__(
        self,
        ligand_code: str,
        working_dir: Optional[str | Path] = None,
        charge_method: str = "bcc",
        atom_type: str = "gaff",
        net_charge: Optional[int] = None,
        multiplicity: int = 1,
        force_field_source: str = "oldff/leaprc.ff99SB",
        verbose: bool = False,
    ) -> None:
        self.ligand_code        = ligand_code
        self.working_dir        = Path(working_dir) if working_dir else Path.cwd()
        self.charge_method      = charge_method
        self.atom_type          = atom_type
        self.net_charge         = net_charge
        self.multiplicity       = multiplicity
        self.force_field_source = force_field_source
        self.verbose            = verbose

        # Derived file paths — all relative to working_dir
        self.input_pdb      = self.working_dir / f"{ligand_code}.pdb"
        self.protonated_pdb = self.working_dir / f"{ligand_code}_h.pdb"
        self.mol2_file      = self.working_dir / f"{ligand_code}.mol2"
        self.frcmod_file    = self.working_dir / f"{ligand_code}.frcmod"
        self.lib_file       = self.working_dir / f"{ligand_code}.lib"
        self.prmtop_file    = self.working_dir / f"{ligand_code}.prmtop"
        self.rst7_file      = self.working_dir / f"{ligand_code}.rst7"
        self.tleap_input    = self.working_dir / "tleap.in"
        self.sqm_out        = self.working_dir / "sqm.out"

        # Output directory created by ACPYPE
        self.output_dir = self.working_dir / f"{ligand_code}.amb2gmx"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Execute the full parameterisation pipeline.

        Returns
        -------
        Path
            Path to the ``{ligand_code}.amb2gmx/`` output directory.

        Raises
        ------
        FileNotFoundError
            If ``{ligand_code}.pdb`` does not exist in ``working_dir``.
        RuntimeError
            If any pipeline step fails or required tools are missing.
        """
        self._check_dependencies()
        self._check_input()

        self._step_protonate()
        self._step_clean_conect()
        self._step_antechamber()
        self._step_check_sqm()
        self._step_parmchk2()
        self._step_tleap()
        self._step_acpype()

        print(f"✓  Parameterisation complete → {self.output_dir}")
        return self.output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_dependencies(self) -> None:
        missing = check_dependencies()
        if missing:
            raise RuntimeError(
                "The following executables are required but not found on PATH:\n"
                + "\n".join(f"  • {exe}" for exe in missing)
                + "\n\nSee LIGAND_PARAMETERISATION.md for installation instructions."
            )

    def _check_input(self) -> None:
        if not self.input_pdb.exists():
            raise FileNotFoundError(
                f"Input PDB not found: {self.input_pdb}\n"
                f"Expected a file named '{self.ligand_code}.pdb' in {self.working_dir}"
            )

    def _run(self, command: list[str], label: str, stdin: Optional[str] = None) -> str:
        """
        Run a subprocess, capture output, and raise on failure.

        Returns
        -------
        str
            Combined stdout of the process.
        """
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=not self.verbose,
            text=True,
            cwd=str(self.working_dir),
        )
        if result.returncode != 0:
            stderr = result.stderr or "(no stderr)"
            raise RuntimeError(
                f"✗  {label} failed (exit {result.returncode})\n"
                f"   Command : {' '.join(command)}\n"
                f"   stderr  : {stderr.strip()}"
            )
        print(f"✓  {label}")
        return result.stdout or ""

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_protonate(self) -> None:
        """
        Step 1: Add hydrogen atoms with Open Babel.

        Open Babel determines protonation state at physiological pH
        (approximately pH 7.4) by default.  For ligands with unusual
        pKa values or a specific pH of interest, consider pre-protonating
        with a dedicated tool (e.g. Dimorphite-DL, ChemAxon Marvin) and
        skipping this step.

        Output: ``{ligand_code}_h.pdb``
        """
        self._run(
            ["obabel",
             str(self.input_pdb),
             "-O", str(self.protonated_pdb),
             "-h"],
            label=f"obabel (protonate {self.ligand_code})",
        )

    def _step_clean_conect(self) -> None:
        """
        Step 2: Strip CONECT records from the protonated PDB.

        CONECT records specify covalent connectivity using PDB atom
        serial numbers.  ``antechamber`` derives connectivity from
        3-D coordinates and atom types instead, and CONECT records from
        Open Babel can cause atom-numbering mismatches that make
        ``antechamber`` fail.  Removing them is always safe for small
        molecules being parameterised with GAFF.

        Output: ``{ligand_code}_h.pdb`` (edited in-place)
        """
        lines = self.protonated_pdb.read_text().splitlines(keepends=True)
        cleaned = [l for l in lines if not l.startswith("CONECT")]
        self.protonated_pdb.write_text("".join(cleaned))
        removed = len(lines) - len(cleaned)
        print(f"✓  Stripped {removed} CONECT record(s) from {self.protonated_pdb.name}")

    def _step_antechamber(self) -> None:
        """
        Step 3: Generate AM1-BCC partial charges and a MOL2 file.

        ``antechamber`` drives a semi-empirical QM calculation (SQM,
        part of AmberTools) to compute bond-charge corrections on top
        of AM1 charges.  The calculation writes progress to ``sqm.out``.

        This is the slowest step for large ligands (> ~50 heavy atoms).
        Runtime scales roughly quadratically with the number of heavy
        atoms.

        Output: ``{ligand_code}.mol2``, ``sqm.out``
        """
        command = [
            "antechamber",
            "-i",  str(self.protonated_pdb),
            "-fi", "pdb",
            "-o",  str(self.mol2_file),
            "-fo", "mol2",
            "-c",  self.charge_method,
            "-at", self.atom_type,
            "-s",  "2",   # verbosity: 2 = standard
        ]
        if self.net_charge is not None:
            command += ["-nc", str(self.net_charge)]
        if self.multiplicity != 1:
            command += ["-m", str(self.multiplicity)]

        self._run(command, label=f"antechamber ({self.charge_method} charges)")

    def _step_check_sqm(self) -> None:
        """
        Step 4: Verify the SQM calculation completed successfully.

        Reads the last four lines of ``sqm.out`` and looks for the
        phrase "Calculation Completed".  Raises ``RuntimeError`` if
        the phrase is absent, indicating that the semi-empirical QM
        job did not converge or was interrupted.

        This mirrors the ``tail -n 4 sqm.out | grep`` check in the
        original shell script.
        """
        if not self.sqm_out.exists():
            raise RuntimeError(
                f"sqm.out not found at {self.sqm_out}.  "
                "The antechamber step may have failed silently."
            )

        tail = self.sqm_out.read_text().splitlines()[-4:]
        if not any("Calculation Completed" in line for line in tail):
            raise RuntimeError(
                f"SQM calculation did not complete successfully.\n"
                f"Check {self.sqm_out} for errors.\n"
                f"Last 4 lines:\n"
                + "\n".join(f"  {l}" for l in tail)
            )
        print(f"✓  SQM calculation completed successfully")

    def _step_parmchk2(self) -> None:
        """
        Step 5: Check for missing GAFF parameters.

        ``parmchk2`` reads the MOL2 file and compares each atom and
        bond type against the GAFF parameter database.  Any parameters
        that are absent or ambiguous are written to a FRCMOD supplement
        file using estimated values (from Lennard-Jones fitting or
        torsion analogy).

        Always inspect the FRCMOD file for ``ATTN, need revision``
        comments — these indicate parameters that could not be estimated
        reliably and may require QM-derived values.

        Output: ``{ligand_code}.frcmod``
        """
        self._run(
            ["parmchk2",
             "-i", str(self.mol2_file),
             "-f", "mol2",
             "-o", str(self.frcmod_file)],
            label="parmchk2 (parameter check)",
        )

    def _step_tleap(self) -> None:
        """
        Step 6: Build Amber PRMTOP/RST7 topology files with tleap.

        Writes a ``tleap.in`` input script and runs ``tleap``.  The
        input loads the FF99SB protein force field (for the tleap
        ``check`` command) and GAFF for the ligand, loads the MOL2
        geometry and FRCMOD parameters, writes a library file, and
        saves the Amber topology (PRMTOP) and coordinate (RST7) files.

        Output: ``tleap.in``, ``{ligand_code}.lib``,
        ``{ligand_code}.prmtop``, ``{ligand_code}.rst7``
        """
        tleap_script = textwrap.dedent(f"""\
            source {self.force_field_source}
            source leaprc.gaff
            {self.ligand_code} = loadmol2 {self.mol2_file.name}
            check {self.ligand_code}
            loadamberparams {self.frcmod_file.name}
            saveoff {self.ligand_code} {self.lib_file.name}
            saveamberparm {self.ligand_code} {self.prmtop_file.name} {self.rst7_file.name}
            quit
        """)

        self.tleap_input.write_text(tleap_script)
        self._run(
            ["tleap", "-f", str(self.tleap_input)],
            label="tleap (build Amber topology)",
        )

    def _step_acpype(self) -> None:
        """
        Step 7: Convert Amber topology to GROMACS format with ACPYPE.

        ACPYPE reads the Amber PRMTOP and RST7 files and writes a
        GROMACS-compatible topology (``.top``), structure (``.gro``),
        and position-restraint (``.itp``) file into a new directory
        named ``{ligand_code}.amb2gmx/``.

        Output directory: ``{ligand_code}.amb2gmx/``
        Key output files:
            ``{ligand_code}_GMX.top``     – GROMACS topology
            ``{ligand_code}_GMX.gro``     – GROMACS structure
            ``posre_{ligand_code}.itp``   – position restraints
        """
        self._run(
            ["acpype",
             "-p", str(self.prmtop_file),
             "-x", str(self.rst7_file)],
            label="acpype (Amber → GROMACS conversion)",
        )
