"""
utils/structure_io.py
=====================
Structure format conversion and cleaning utilities for simulation preparation.

Provides a single public function — :func:`prepare_structure` — that
accepts a protein or ligand structure in any of the supported input
formats and writes a clean, GROMACS-ready PDB file.

Supported input formats
-----------------------

Protein / complex
~~~~~~~~~~~~~~~~~
``".pdb"``
    Standard PDB.  Passed through with optional cleaning.
``".cif"`` / ``".mmcif"``
    mmCIF / PDBx format (default for all RCSB depositions since 2019,
    and for all AlphaFold models).  Converted via ``gemmi``.
``".gro"``
    GROMACS coordinate format.  Converted via GROMACS ``gmx editconf``
    to PDB.

Ligand only
~~~~~~~~~~~
``".mol2"``
    Tripos MOL2 — common output from Glide, GOLD, and AutoDock Vina
    ``--out-format mol2`` runs.
``".sdf"`` / ``".mol"``
    MDL / SDF format — standard from PubChem, ChEMBL, and most
    virtual-screening pipelines.
``"smiles:<string>"``
    Inline SMILES string (prefix ``"smiles:"``).  3-D coordinates are
    generated with RDKit's ETKDG conformer generator and energy-
    minimised with the MMFF94 force field.

Dependencies
------------
``gemmi``    — mmCIF conversion (``pip install gemmi`` or
               ``conda install -c conda-forge gemmi``)
``rdkit``    — MOL2, SDF, SMILES handling (conda only:
               ``conda install -c conda-forge rdkit``)

Both are optional for users who work exclusively with PDB input.
``ImportError`` is raised with a helpful message if a required library
is not installed for the requested format.

Example usage
-------------
    from utils.structure_io import prepare_structure

    # AlphaFold CIF → clean PDB
    pdb_path = prepare_structure(
        "AF-P04637-F1-model_v4.cif",
        output_name="p53",
        work_dir="/data/simulations/p53_apo",
    )

    # SDF ligand → PDB
    lig_path = prepare_structure(
        "inhibitor.sdf",
        output_name="INH",
        work_dir="/data/simulations/p53_complex",
        ligand_code="INH",
    )

    # SMILES → PDB (no input file needed)
    lig_path = prepare_structure(
        "smiles:CC(=O)Nc1ccc(O)cc1",  # paracetamol
        output_name="PAR",
        work_dir="/data/simulations/p53_complex",
        ligand_code="PAR",
    )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_structure(
    input_path: str,
    output_name: str,
    work_dir: Optional[str | Path] = None,
    ligand_code: Optional[str] = None,
    remove_hetatm: bool = True,
    remove_waters: bool = True,
    keep_residues: Optional[list[str]] = None,
    gmx_executable: str = "gmx",
) -> Path:
    """
    Convert any supported structure format to a clean PDB file.

    Parameters
    ----------
    input_path : str
        Path to the input file, or an inline SMILES string prefixed with
        ``"smiles:"`` (e.g. ``"smiles:CC(=O)Nc1ccc(O)cc1"``).
    output_name : str
        Stem of the output PDB file (without ``.pdb``).  The file is
        written as ``{work_dir}/{output_name}.pdb``.
    work_dir : str or Path, optional
        Directory where the output PDB is written.  Defaults to the
        current working directory.
    ligand_code : str, optional
        Three-letter residue code to use in the output PDB for ligand
        atoms.  Required when converting MOL2, SDF, or SMILES inputs.
    remove_hetatm : bool, optional
        Strip ``HETATM`` records from PDB / mmCIF protein structures.
        Defaults to ``True``.  Set to ``False`` if your structure
        contains a co-crystallised ligand you wish to keep for
        reference.
    remove_waters : bool, optional
        Strip water molecules (``HOH``, ``WAT``, ``TIP``) from the
        output.  Defaults to ``True``.
    keep_residues : list[str], optional
        Residue names to keep even when ``remove_hetatm=True``.
        Useful for retaining metal ions (e.g. ``["ZN", "MG"]``) or
        specific co-factors.  Defaults to ``None`` (nothing kept beyond
        protein ``ATOM`` records).
    gmx_executable : str, optional
        GROMACS binary name or full path.  Used only for ``.gro`` input.
        Defaults to ``"gmx"``.

    Returns
    -------
    Path
        Path to the written ``{output_name}.pdb`` file.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` is not a SMILES string and the file does not
        exist.
    ValueError
        If the file extension is not recognised.
    ImportError
        If a required library (``gemmi`` or ``rdkit``) is not installed
        for the requested format.
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    keep_residues = keep_residues or []
    out_pdb = work_dir / f"{output_name}.pdb"

    # ---- Route by format -----------------------------------------------
    if input_path.startswith("smiles:"):
        smiles = input_path[len("smiles:"):]
        _smiles_to_pdb(smiles, out_pdb, ligand_code or output_name[:3].upper())

    else:
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"Input structure not found: {src}")

        suffix = src.suffix.lower()

        if suffix == ".pdb":
            _clean_pdb(src, out_pdb, remove_hetatm, remove_waters, keep_residues)

        elif suffix in (".cif", ".mmcif"):
            _cif_to_pdb(src, out_pdb, remove_hetatm, remove_waters, keep_residues)

        elif suffix == ".gro":
            _gro_to_pdb(src, out_pdb, gmx_executable)

        elif suffix in (".mol2",):
            if ligand_code is None:
                raise ValueError("ligand_code is required for MOL2 input")
            _mol2_to_pdb(src, out_pdb, ligand_code)

        elif suffix in (".sdf", ".mol"):
            if ligand_code is None:
                raise ValueError("ligand_code is required for SDF/MOL input")
            _sdf_to_pdb(src, out_pdb, ligand_code)

        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'.\n"
                f"Supported: .pdb, .cif, .mmcif, .gro, .mol2, .sdf, .mol, "
                f"or 'smiles:<string>'"
            )

    print(f"✓  Structure prepared → {out_pdb}")
    return out_pdb


# ---------------------------------------------------------------------------
# Internal converters
# ---------------------------------------------------------------------------

def _clean_pdb(
    src: Path,
    dest: Path,
    remove_hetatm: bool,
    remove_waters: bool,
    keep_residues: list[str],
) -> None:
    """
    Filter a PDB file: keep ATOM records, optionally strip HETATM and
    water, and always strip MODEL/ENDMDL (keep first model only),
    ANISOU (redundant with ATOM), and alternate-location indicators
    beyond the first.

    The first alternate location (altLoc 'A' or '1') is kept and the
    altLoc column is blanked to produce a single-model PDB that
    GROMACS can handle without complaints.
    """
    _WATER_NAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL"}

    lines_out: list[str] = []
    seen_model = False
    in_model   = False

    with src.open() as fh:
        for line in fh:
            record = line[:6].strip()

            # Only keep the first MODEL block
            if record == "MODEL":
                if seen_model:
                    break       # stop reading at second MODEL
                seen_model = True
                in_model   = True
                continue        # don't write MODEL card itself
            if record == "ENDMDL":
                in_model = False
                continue

            # Skip ANISOU — redundant and sometimes confuses pdb2gmx
            if record == "ANISOU":
                continue

            if record in ("ATOM", "HETATM"):
                res_name = line[17:20].strip()
                alt_loc  = line[16]

                # Skip non-first alternate locations
                if alt_loc not in (" ", "A", "1", ""):
                    continue
                # Blank the altLoc column so downstream tools don't see it
                line = line[:16] + " " + line[17:]

                if record == "HETATM":
                    is_water = res_name in _WATER_NAMES
                    if is_water and remove_waters:
                        continue
                    if not is_water and remove_hetatm and res_name not in keep_residues:
                        continue

            lines_out.append(line)

    dest.write_text("".join(lines_out))
    print(f"  PDB cleaned: {len(lines_out)} records written")


def _cif_to_pdb(
    src: Path,
    dest: Path,
    remove_hetatm: bool,
    remove_waters: bool,
    keep_residues: list[str],
) -> None:
    """
    Convert an mmCIF file to PDB using ``gemmi``, then apply the same
    cleaning as :func:`_clean_pdb`.

    ``gemmi`` is required for this path.  It reads the full mmCIF
    format (including PDBx extensions) and writes a standards-compliant
    PDB file, correctly handling chain IDs, insertion codes, and
    non-standard residue names.
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "The 'gemmi' package is required to read mmCIF files.\n"
            "Install it with:  conda install -c conda-forge gemmi\n"
            "             or:  pip install gemmi"
        )

    structure = gemmi.read_structure(str(src))

    # gemmi can write PDB directly; write to a temp file then clean
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        structure.write_pdb(str(tmp_path))
        _clean_pdb(tmp_path, dest, remove_hetatm, remove_waters, keep_residues)
    finally:
        tmp_path.unlink(missing_ok=True)


def _gro_to_pdb(src: Path, dest: Path, gmx_executable: str) -> None:
    """
    Convert a GROMACS ``.gro`` file to PDB using ``gmx editconf``.
    """
    result = subprocess.run(
        [gmx_executable, "editconf", "-f", str(src), "-o", str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gmx editconf failed converting {src.name} to PDB.\n"
            f"stderr: {result.stderr.strip()}"
        )


def _mol2_to_pdb(src: Path, dest: Path, ligand_code: str) -> None:
    """
    Convert a MOL2 file to PDB using RDKit, setting the residue name
    to ``ligand_code``.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit is required to read MOL2 files.\n"
            "Install via conda:  conda install -c conda-forge rdkit"
        )

    mol = Chem.MolFromMol2File(str(src), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse MOL2 file: {src}")

    mol = _set_residue_name(mol, ligand_code)
    Chem.MolToPDBFile(mol, str(dest))


def _sdf_to_pdb(src: Path, dest: Path, ligand_code: str) -> None:
    """
    Convert an SDF / MOL file to PDB using RDKit, setting the residue
    name to ``ligand_code``.  Only the first molecule in the SDF is
    used.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit is required to read SDF/MOL files.\n"
            "Install via conda:  conda install -c conda-forge rdkit"
        )

    supplier = Chem.SDMolSupplier(str(src), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"RDKit could not parse any molecule from: {src}")

    mol = _set_residue_name(mol, ligand_code)
    Chem.MolToPDBFile(mol, str(dest))


def _smiles_to_pdb(smiles: str, dest: Path, ligand_code: str) -> None:
    """
    Generate a 3-D structure from a SMILES string using RDKit's ETKDG
    conformer generator, minimise with MMFF94, and write a PDB.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    dest : Path
        Output PDB path.
    ligand_code : str
        Three-letter residue name written into the PDB.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed, or if 3-D embedding
        fails (common for very large or highly constrained molecules).
    ImportError
        If RDKit is not installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit is required for SMILES-to-PDB conversion.\n"
            "Install via conda:  conda install -c conda-forge rdkit"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: '{smiles}'")

    # Add explicit hydrogens before 3-D embedding
    mol = Chem.AddHs(mol)

    # ETKDG: distance-geometry conformer generation with knowledge-based
    # torsion angle preferences — more reliable than random embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)

    if result == -1:
        raise ValueError(
            f"RDKit ETKDG conformer generation failed for SMILES: '{smiles}'.\n"
            "This can happen for very large, highly constrained, or unusual "
            "molecules.  Try providing a pre-built .sdf or .mol2 file instead."
        )

    # MMFF94 energy minimisation for a reasonable starting geometry
    ff_result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
    if ff_result == -1:
        print(
            f"  Warning: MMFF94 minimisation failed for '{ligand_code}'. "
            "Coordinates from ETKDG will be used without minimisation."
        )

    mol = _set_residue_name(mol, ligand_code)
    Chem.MolToPDBFile(mol, str(dest))


# ---------------------------------------------------------------------------
# RDKit helper
# ---------------------------------------------------------------------------

def _set_residue_name(mol: Any, residue_name: str) -> Any:
    """
    Set the PDB residue name on every atom in an RDKit molecule.

    RDKit writes residue information from ``AtomPDBResidueInfo`` objects
    attached to each atom.  If no residue info exists, a new one is
    created.  This ensures the output PDB has the correct three-letter
    residue code for downstream GROMACS processing.
    """
    from rdkit.Chem import AllChem

    residue_name = residue_name[:3].upper()   # PDB columns 18-20 are 3 chars

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            element = atom.GetSymbol()
            atom_name = f"{element:<2}{atom.GetIdx() + 1:<2}"[:4]
            new_info = AllChem.AtomPDBResidueInfo(
                atomName=atom_name,
                residueName=residue_name,
                residueNumber=1,
                chainId="A",
                isHeteroAtom=True,
            )
            atom.SetMonomerInfo(new_info)
        else:
            info.SetResidueName(residue_name)
    return mol
