"""
base.py
=======
Abstract base class for all GROMACS simulation preparation workflows.

Subclasses
----------
ApoSimPrepper   – apo (protein-only) simulations
CplxSimPrepper  – protein–ligand complex simulations
MixMDPrepper    – mixed-solvent (MixMD) simulations

Config sources
--------------
Every subclass accepts configuration from two equivalent sources:

1. **Keyword arguments** — passed directly at construction time::

       sim = ApoSimPrepper(protein_name="hsp90", sim_len=100, ...)

2. **YAML / JSON config file** — loaded by :func:`sim_prep.config.load_config`,
   which parses the file and unpacks it as kwargs::

       sim = load_config("apo.yaml")

Both routes produce an identical internal ``self.config`` dict.  YAML
``null`` values for optional keys are normalised to Python ``None`` and
then replaced with their defaults during :meth:`assign_attributes`, so
commented-out optional fields in a config file are handled correctly.

Common pipeline
---------------
Every subclass runs the following shared steps in order:

    1.  validate_config()       ← abstract; each subclass validates its own params
    2.  assign_attributes()     ← populate instance attributes from validated config
    3.  copy_config_files()     ← copy MDP templates to the working directory
    4.  update_config_files()   ← patch nsteps / tc-grps / etc. into MDP files
    5.  clean_pdb_file()        ← strip non-ATOM records from the input PDB
    6.  protein_pdb2gmx()       ← gmx pdb2gmx → processed .gro + topol.top
    7.  set_new_box()           ← gmx editconf → simulation box
    8.  solvate()               ← gmx solvate + genion → solv_ions.gro
    9.  minimise_system()       ← gmx grompp + mdrun → energy minimisation
    10. nvt_equilibration()     ← constant-volume equilibration
    11. npt_equilibration()     ← constant-pressure equilibration
    12. production_run()        ← MD production run
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants used for validation in subclasses
# ---------------------------------------------------------------------------

VALID_BOX_SHAPES: tuple[str, ...] = (
    "cubic", "triclinic", "dodecahedron", "octahedron"
)

VALID_POS_IONS: tuple[str, ...] = ("NA", "K", "MG", "CA", "ZN")
VALID_NEG_IONS: tuple[str, ...] = ("CL",)

LIMITS: dict[str, tuple[float, float]] = {
    "sim_len": (0.001, 10_000.0),   # nanoseconds
    "bx_dim":  (0.5,   5.0),        # nanometres padding
}

#: Femtoseconds per step (standard GROMACS 2-fs timestep)
_FS_PER_STEP: int = 2
#: Steps per nanosecond at 2 fs/step
_STEPS_PER_NS: int = 500_000


# ---------------------------------------------------------------------------
# Helper: normalise a raw config dict from YAML/kwargs
# ---------------------------------------------------------------------------

def _normalise_config(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Return a clean copy of *raw* with YAML ``null`` values (``None``)
    removed for optional keys that have class-level defaults.

    YAML produces ``None`` for commented-out lines such as::

        # net_charge: -1

    which would silently override Python default argument values if left in
    the dict.  Removing ``None``-valued keys lets ``.get("key", default)``
    return the proper default everywhere in the codebase.

    Required keys (``protein_name``, ``sim_len``, etc.) are left untouched
    so that ``validate_config()`` can catch them as missing.
    """
    #: Keys that are legitimately ``None`` (i.e. the user explicitly set null
    #: to mean "use the default").  All others are passed through as-is.
    _NULLABLE_OPTIONAL_KEYS = {
        "work_dir", "gmx_executable", "index_file",
        "net_charge", "atom_type", "charge_method",
        "protein_input", "ligand_input", "keep_residues",
        "net_charges", "begin", "end", "dt",
        "pos_ion", "neg_ion",
        "param_ligand_name", "remove_ligands",
    }
    return {
        k: v for k, v in raw.items()
        if not (k in _NULLABLE_OPTIONAL_KEYS and v is None)
    }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SimulationPrepper(ABC):
    """
    Abstract base class orchestrating GROMACS simulation preparation.

    Accepts configuration as keyword arguments (direct construction) or
    from a parsed YAML/JSON dict (via :func:`sim_prep.config.load_config`).
    Both routes are equivalent — the ``self.config`` dict is the single
    source of truth for all parameter access.

    Parameters
    ----------
    protein_name : str
        Stem of the input PDB file (without ``.pdb`` extension).
    sim_len : float
        Production run length in nanoseconds.
    bx_dim : float
        Padding added to the protein diameter when building the box (nm).
    bx_shp : str
        Box shape.  One of ``"cubic"``, ``"triclinic"``,
        ``"dodecahedron"``, ``"octahedron"``.
    md_name : str
        Stem used for all production-run output files.
    pos_ion : str, optional
        Positive counter-ion for neutralisation.  Defaults to ``"NA"``.
    neg_ion : str, optional
        Negative counter-ion for neutralisation.  Defaults to ``"CL"``.
    work_dir : str or Path, optional
        Working directory.  Defaults to the current working directory.
    gmx_executable : str, optional
        Name or full path of the GROMACS binary.  Defaults to ``"gmx"``.
    index_file : str, optional
        Path to a custom ``.ndx`` index file.  Defaults to ``None``.
    **kwargs
        Additional keyword arguments stored in ``self.config`` and
        available to subclasses.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Normalise None-valued optional keys produced by YAML null values
        self.config: dict[str, Any] = _normalise_config(kwargs)

        # Resolve working directory
        work_dir = self.config.get("work_dir")
        self.working_dir: Path = Path(work_dir) if work_dir else Path.cwd()

        # Allow the gmx executable to be overridden (e.g. on HPC with modules)
        self.gmx: str = self.config.get("gmx_executable", "gmx")

        # Optional index file
        self.index_file: Optional[str] = self.config.get("index_file")

        # Locate the package root so subclasses can resolve config paths
        self.script_directory: Path = Path(__file__).resolve().parent

        # These will be populated by assign_attributes() after validation
        self.protein_name: Optional[str] = None
        self.sim_length:   Optional[float] = None
        self.box_dim:      Optional[str] = None    # kept as str for gmx editconf -d
        self.box_type:     Optional[str] = None
        self.md_name:      Optional[str] = None
        self.pos_ion:      Optional[str] = None
        self.neg_ion:      Optional[str] = None
        self.topology:     Optional[Path] = None

        # Subclasses assign their own mdp_files dict and config_dir
        self.mdp_files:  Optional[dict[str, str]] = None
        self.config_dir: Optional[Path] = None

        # Force-field and genion interactive selections
        # Defaults: AMBER99SB-ILDN (6) + TIP3P (1) — override in subclass
        self.ff_selection: str = "6\n1\n"
        self.genion_sele:  str = "13"   # SOL in a standard protein-water system

        # Checkpoint flags (used by CheckpointMixin if inherited)
        self.cleaned_pdb:  bool = False
        self.em_complete:  bool = False
        self.nvt_complete: bool = False
        self.npt_complete: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate ``self.config`` for the specific simulation type.
        Raise :exc:`ValueError` with a descriptive bullet-point message
        listing every problem found (not just the first).
        """

    @abstractmethod
    def update_config_files(self) -> None:
        """
        Patch MDP files that have been copied to the working directory
        with simulation-type-specific parameters (e.g. nsteps, tc-grps).
        """

    # ------------------------------------------------------------------
    # Shared: index file helper
    # ------------------------------------------------------------------

    def _ndx_flag(self) -> list[str]:
        """Return ``['-n', index_file]`` when an index file is set, else ``[]``."""
        return ["-n", self.index_file] if self.index_file else []

    # ------------------------------------------------------------------
    # Shared: attribute population
    # ------------------------------------------------------------------

    def assign_attributes(self) -> None:
        """
        Populate typed instance attributes from the validated config dict.

        Call this immediately after :meth:`validate_config`.  Safe to
        call from a YAML-loaded config — ``None`` values for optional
        fields have already been stripped by :func:`_normalise_config`
        so ``.get(key, default)`` always returns the correct default.
        """
        cfg = self.config
        self.protein_name = cfg["protein_name"]
        self.sim_length   = float(cfg["sim_len"])
        # box_dim is stored as str because gmx editconf -d expects a string
        self.box_dim      = str(cfg.get("bx_dim", "1.0"))
        self.box_type     = cfg.get("bx_shp", "dodecahedron")
        self.md_name      = cfg.get("md_name", "md_production")
        self.pos_ion      = cfg.get("pos_ion", "NA")
        self.neg_ion      = cfg.get("neg_ion", "CL")

    # ------------------------------------------------------------------
    # Shared: MDP file management
    # ------------------------------------------------------------------

    def copy_config_files(self) -> None:
        """
        Copy all MDP template files listed in ``self.mdp_files`` from
        ``self.config_dir`` to ``self.working_dir``.

        Raises
        ------
        RuntimeError
            If ``mdp_files`` or ``config_dir`` have not been set by the
            subclass ``__init__``.
        FileNotFoundError
            If a template file is missing from ``config_dir``.
        """
        if self.mdp_files is None or self.config_dir is None:
            raise RuntimeError(
                "mdp_files and config_dir must be set before calling "
                "copy_config_files().  This is done in the subclass __init__."
            )
        for _key, base in self.mdp_files.items():
            filename = f"{base}.mdp"
            src  = self.config_dir / filename
            dest = self.working_dir / filename
            if not src.exists():
                raise FileNotFoundError(
                    f"MDP template not found: {src}\n"
                    f"Populate config/gmx/{self.config_dir.name}/ with your "
                    f"MDP templates before running."
                )
            shutil.copy(src, dest)
            print(f"✓  Copied {filename} → {self.working_dir}")

    def _calc_nsteps(self) -> int:
        """Return the number of integration steps for the production run."""
        return int(self.sim_length * _STEPS_PER_NS)

    def _patch_nsteps(self, mdp_path: Path) -> None:
        """
        Replace the ``nsteps`` line in *mdp_path* with the value derived
        from ``self.sim_length``.  The file is edited in-place.

        Parameters
        ----------
        mdp_path : Path
            Path to an MDP file that has already been copied to the
            working directory.
        """
        nsteps = self._calc_nsteps()
        lines  = mdp_path.read_text().splitlines(keepends=True)
        patched: list[str] = []
        for line in lines:
            if line.startswith("nsteps"):
                duration_ps = self.sim_length * 1000
                line = (
                    f"nsteps                  = {nsteps}"
                    f" ; {_FS_PER_STEP} * {nsteps} = {duration_ps:.0f} ps"
                    f" ({self.sim_length} ns)\n"
                )
                print(
                    f"✓  nsteps → {nsteps}  "
                    f"({self.sim_length} ns at {_FS_PER_STEP} fs/step)"
                )
            patched.append(line)
        mdp_path.write_text("".join(patched))

    # ------------------------------------------------------------------
    # Shared: PDB cleaning
    # ------------------------------------------------------------------

    def clean_pdb_file(self) -> None:
        """
        Strip all non-ATOM records from the input PDB and write
        ``{protein_name}_clean.pdb`` to the working directory.

        This is a lightweight clean for systems whose PDB is already
        well-formed.  For structures from the RCSB or AlphaFold (which
        may contain missing residues, non-standard residues, or alternate
        conformations), use :func:`utils.structure_io.prepare_structure`
        instead, which handles mmCIF conversion and more thorough cleaning.

        Output files
        ------------
        ``{protein_name}_clean.pdb``
            PDB containing only ``ATOM`` records.
        """
        src  = self.working_dir / f"{self.protein_name}.pdb"
        dest = self.working_dir / f"{self.protein_name}_clean.pdb"

        if not src.exists():
            raise FileNotFoundError(
                f"Input PDB not found: {src}\n"
                f"Expected '{self.protein_name}.pdb' in {self.working_dir}"
            )

        with src.open() as fh:
            atom_lines = [l for l in fh if l.startswith("ATOM")]

        with dest.open("w") as fh:
            fh.writelines(atom_lines)

        self.cleaned_pdb = True
        print(f"✓  Cleaned PDB → {dest.name}  ({len(atom_lines)} ATOM records)")

    # ------------------------------------------------------------------
    # Shared: subprocess runner
    # ------------------------------------------------------------------

    def _run(
        self,
        command: list[str],
        label: str,
        stdin: Optional[str] = None,
    ) -> None:
        """
        Execute a GROMACS subprocess with uniform error handling.

        Parameters
        ----------
        command : list[str]
            Full command list including the ``gmx`` binary.
        label : str
            Human-readable label printed on success / failure.
        stdin : str, optional
            Text piped to the process's standard input (interactive
            group selections).

        Raises
        ------
        subprocess.CalledProcessError
            Re-raised after printing the failed command and error output,
            so the caller's traceback points to the failing pipeline step.
        """
        try:
            subprocess.run(
                command,
                input=stdin,
                check=True,
                text=True,
            )
            print(f"✓  {label}")
        except subprocess.CalledProcessError as exc:
            print(f"✗  {label} – command failed")
            print(f"   Command : {' '.join(command)}")
            print(f"   Error   : {exc}")
            raise

    # ------------------------------------------------------------------
    # Shared: GROMACS pipeline steps
    # ------------------------------------------------------------------

    def protein_pdb2gmx(self) -> None:
        """
        Convert the cleaned PDB to GROMACS format using ``gmx pdb2gmx``.

        Output files
        ------------
        ``{protein_name}_processed.gro``, ``topol.top``, ``posre.itp``
        """
        self._run(
            [
                self.gmx, "pdb2gmx",
                "-f", f"{self.protein_name}_clean.pdb",
                "-o", f"{self.protein_name}_processed.gro",
                *self._ndx_flag(),
                "-ter",
                "-ignh",
            ],
            label="pdb2gmx",
            stdin=self.ff_selection,
        )
        self.topology = self.working_dir / "topol.top"

    def set_new_box(self) -> None:
        """
        Define the simulation box using ``gmx editconf``.

        Output files
        ------------
        ``newbox.gro``
        """
        self._run(
            [
                self.gmx, "editconf",
                "-f", f"{self.protein_name}_processed.gro",
                "-d", self.box_dim,
                "-bt", self.box_type,
                "-o", "newbox.gro",
            ],
            label="editconf (set box)",
        )

    def solvate(self) -> None:
        """
        Solvate the system and add counter-ions.

        Steps
        -----
        1. ``gmx solvate`` — fill the box with SPC/E water.
        2. ``gmx grompp`` — pre-process for ion addition.
        3. ``gmx genion`` — replace water with counter-ions.

        Output files
        ------------
        ``solv.gro``, ``ions.tpr``, ``solv_ions.gro``
        """
        ions_mdp = f"{self.mdp_files['ions']}.mdp"

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
            stdin=self.genion_sele,
        )

    def minimise_system(self, maxwarn: Optional[int] = None) -> None:
        """
        Energy-minimise the solvated system.

        Parameters
        ----------
        maxwarn : int, optional
            Passed as ``-maxwarn`` to ``grompp`` when set.

        Output files
        ------------
        ``em.tpr``, ``em.gro``, ``em.edr``, ``em.log``
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
        self._run(
            [self.gmx, "mdrun", "-v", "-deffnm", "em"],
            label="mdrun (EM)",
        )
        self.em_complete = True

    def nvt_equilibration(self, maxwarn: Optional[int] = None) -> None:
        """
        Run NVT (constant volume) equilibration.

        Parameters
        ----------
        maxwarn : int, optional
            Passed as ``-maxwarn`` to ``grompp`` when set.

        Output files
        ------------
        ``nvt.tpr``, ``nvt.gro``, ``nvt.cpt``, ``nvt.edr``, ``nvt.log``
        """
        nvt_mdp = f"{self.mdp_files['nvt']}.mdp"

        grompp_cmd = [
            self.gmx, "grompp",
            "-f", nvt_mdp,
            "-c", "em.gro",
            "-r", "em.gro",
            "-p", "topol.top",
            "-o", "nvt.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        self._run(grompp_cmd, label="grompp (NVT)")
        self._run(
            [self.gmx, "mdrun", "-v", "-deffnm", "nvt"],
            label="mdrun (NVT)",
        )
        self.nvt_complete = True

    def npt_equilibration(self, maxwarn: Optional[int] = None) -> None:
        """
        Run NPT (constant pressure) equilibration.

        Parameters
        ----------
        maxwarn : int, optional
            Passed as ``-maxwarn`` to ``grompp`` when set.

        Output files
        ------------
        ``npt.tpr``, ``npt.gro``, ``npt.cpt``, ``npt.edr``, ``npt.log``
        """
        npt_mdp = f"{self.mdp_files['npt']}.mdp"

        grompp_cmd = [
            self.gmx, "grompp",
            "-f", npt_mdp,
            "-c", "nvt.gro",
            "-t", "nvt.cpt",
            "-r", "nvt.gro",
            "-p", "topol.top",
            "-o", "npt.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        self._run(grompp_cmd, label="grompp (NPT)")
        self._run(
            [self.gmx, "mdrun", "-v", "-deffnm", "npt"],
            label="mdrun (NPT)",
        )
        self.npt_complete = True

    def production_run(
        self,
        maxwarn: Optional[int] = None,
        extra_mdrun_args: Optional[list[str]] = None,
    ) -> None:
        """
        Run the MD production simulation.

        Parameters
        ----------
        maxwarn : int, optional
            Passed as ``-maxwarn`` to ``grompp`` when set.
        extra_mdrun_args : list[str], optional
            Additional arguments appended to the ``mdrun`` command, e.g.
            ``["-ntmpi", "1", "-ntomp", "8"]`` for GPU offloading.

        Output files
        ------------
        ``{md_name}.tpr``, ``{md_name}.xtc``, ``{md_name}.gro``,
        ``{md_name}.edr``, ``{md_name}.log``
        """
        md_mdp = f"{self.mdp_files['md']}.mdp"

        grompp_cmd = [
            self.gmx, "grompp",
            "-f", md_mdp,
            "-c", "npt.gro",
            "-t", "npt.cpt",
            "-p", "topol.top",
            "-o", f"{self.md_name}.tpr",
        ]
        if maxwarn is not None:
            grompp_cmd += ["-maxwarn", str(maxwarn)]

        mdrun_cmd = [
            self.gmx, "mdrun",
            "-s", f"{self.md_name}.tpr",
            "-v",
            "-deffnm", self.md_name,
        ]
        if extra_mdrun_args:
            mdrun_cmd += extra_mdrun_args

        self._run(grompp_cmd, label="grompp (production)")
        self._run(mdrun_cmd,  label=f"mdrun (production – {self.sim_length} ns)")

    def update_topology_molecules(
        self, molecule_name: str, molecule_count: int
    ) -> None:
        """
        Append a molecule entry to the ``[ molecules ]`` section of
        ``topol.top``.

        Parameters
        ----------
        molecule_name : str
            Residue / molecule name as it appears in the topology.
        molecule_count : int
            Number of copies to add.

        Raises
        ------
        FileNotFoundError
            If ``topol.top`` does not exist (``protein_pdb2gmx()`` not
            yet called).
        """
        if self.topology is None or not self.topology.exists():
            raise FileNotFoundError(
                f"Topology file not found at '{self.topology}'.  "
                "Run protein_pdb2gmx() first."
            )
        with self.topology.open("a") as fh:
            fh.write(f"{molecule_name}\t\t{molecule_count}\n")
        print(f"✓  Topology updated: {molecule_name} × {molecule_count}")
