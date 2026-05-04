"""
tests/test_suite.py
===================
Test suite for the mdanalysis-gromacs toolkit.

Tests are organised into three groups:

    Unit tests       — pure-Python functions with no external dependencies
    Integration tests — tests that require GROMACS on PATH (skipped otherwise)
    Config tests     — YAML / JSON config loading and template generation

Run with::

    pytest tests/test_suite.py -v

Run only unit tests (no GROMACS required)::

    pytest tests/test_suite.py -v -m "not integration"

Run with coverage report::

    pytest tests/test_suite.py --cov=. --cov-report=term-missing
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gmx_available() -> bool:
    return shutil.which("gmx") is not None

def _has_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

requires_gmx   = pytest.mark.skipif(not _gmx_available(),    reason="GROMACS not on PATH")
requires_gemmi = pytest.mark.skipif(not _has_package("gemmi"), reason="gemmi not installed")
requires_rdkit = pytest.mark.skipif(not _has_package("rdkit"), reason="rdkit not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a fresh temporary directory for each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_xvg(tmp_dir: Path) -> Path:
    """Write a minimal two-column XVG file and return its path."""
    content = """\
# This file was created by gmx rmsf
@ title "RMSF"
@ xaxis label "Residue"
@ yaxis label "RMSF (nm)"
1   0.045
2   0.031
3   0.088
4   0.122
5   0.056
"""
    path = tmp_dir / "test.xvg"
    path.write_text(content)
    return path


@pytest.fixture
def sample_pdb(tmp_dir: Path) -> Path:
    """Write a minimal PDB file (two residues, one HETATM, one water)."""
    content = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.10           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.15           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.12           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.11           O
ATOM      5  N   GLY A   2       4.000   2.000   3.000  1.00  0.09           N
ATOM      6  CA  GLY A   2       5.000   2.000   3.000  1.00  0.10           C
HETATM    7  C1  LIG A 100       6.000   2.000   3.000  1.00  0.20           C
HETATM    8  O1  HOH A 101       7.000   2.000   3.000  1.00  0.30           O
ANISOU    9  N   ALA A   1      100    200    300     10     20     30       N
END
"""
    path = tmp_dir / "test.pdb"
    path.write_text(content)
    return path


@pytest.fixture
def minimal_sim(tmp_dir: Path):
    """Return an ApoSimPrepper instance without calling validate_config."""
    from sim_prep.apo import ApoSimPrepper
    return ApoSimPrepper(
        protein_name="test_protein",
        sim_len=1,
        bx_dim=1.0,
        bx_shp="dodecahedron",
        md_name="md_test",
        pos_ion="NA",
        neg_ion="CL",
        work_dir=str(tmp_dir),
    )


# ===========================================================================
# 1. XVG parser
# ===========================================================================

class TestParseXvg:
    def test_returns_two_arrays(self, sample_xvg: Path):
        from gromacs_analysis import GromacsAnalysis
        x, y = GromacsAnalysis._parse_xvg(str(sample_xvg))
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_correct_length(self, sample_xvg: Path):
        from gromacs_analysis import GromacsAnalysis
        x, y = GromacsAnalysis._parse_xvg(str(sample_xvg))
        assert len(x) == 5
        assert len(y) == 5

    def test_comment_lines_skipped(self, sample_xvg: Path):
        from gromacs_analysis import GromacsAnalysis
        x, y = GromacsAnalysis._parse_xvg(str(sample_xvg))
        # First data point should be residue 1, RMSF 0.045
        assert x[0] == pytest.approx(1.0)
        assert y[0] == pytest.approx(0.045)

    def test_values_correct(self, sample_xvg: Path):
        from gromacs_analysis import GromacsAnalysis
        x, y = GromacsAnalysis._parse_xvg(str(sample_xvg))
        expected_x = [1, 2, 3, 4, 5]
        expected_y = [0.045, 0.031, 0.088, 0.122, 0.056]
        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)


# ===========================================================================
# 2. PDB cleaning (structure_io)
# ===========================================================================

class TestCleanPdb:
    def test_atom_records_kept(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=True,
                   remove_waters=True, keep_residues=[])
        lines = out.read_text().splitlines()
        record_types = {l[:6].strip() for l in lines if l.strip()}
        assert "ATOM" in record_types

    def test_hetatm_removed_by_default(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=True,
                   remove_waters=True, keep_residues=[])
        content = out.read_text()
        assert "HETATM" not in content

    def test_hetatm_kept_when_disabled(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=False,
                   remove_waters=False, keep_residues=[])
        content = out.read_text()
        assert "HETATM" in content

    def test_water_removed(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=False,
                   remove_waters=True, keep_residues=[])
        content = out.read_text()
        assert "HOH" not in content

    def test_keep_residues_overrides_hetatm(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=True,
                   remove_waters=True, keep_residues=["LIG"])
        content = out.read_text()
        assert "LIG" in content

    def test_anisou_removed(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import _clean_pdb
        out = tmp_dir / "clean.pdb"
        _clean_pdb(sample_pdb, out, remove_hetatm=False,
                   remove_waters=False, keep_residues=[])
        content = out.read_text()
        assert "ANISOU" not in content


# ===========================================================================
# 3. prepare_structure routing
# ===========================================================================

class TestPrepareStructureRouting:
    def test_unsupported_format_raises(self, tmp_dir: Path):
        from utils.structure_io import prepare_structure
        fake = tmp_dir / "mol.xyz"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            prepare_structure(str(fake), "out", work_dir=tmp_dir)

    def test_missing_file_raises(self, tmp_dir: Path):
        from utils.structure_io import prepare_structure
        with pytest.raises(FileNotFoundError):
            prepare_structure("nonexistent.pdb", "out", work_dir=tmp_dir)

    def test_mol2_without_ligand_code_raises(self, tmp_dir: Path):
        from utils.structure_io import prepare_structure
        fake = tmp_dir / "lig.mol2"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="ligand_code is required"):
            prepare_structure(str(fake), "out", work_dir=tmp_dir)

    def test_pdb_pass_through(self, sample_pdb: Path, tmp_dir: Path):
        from utils.structure_io import prepare_structure
        out = prepare_structure(
            str(sample_pdb), "clean",
            work_dir=tmp_dir,
            remove_hetatm=True, remove_waters=True,
        )
        assert out.exists()
        assert out.suffix == ".pdb"

    @requires_rdkit
    def test_smiles_generates_pdb(self, tmp_dir: Path):
        from utils.structure_io import prepare_structure
        out = prepare_structure(
            "smiles:CC(=O)Nc1ccc(O)cc1",   # paracetamol
            "PAR",
            work_dir=tmp_dir,
            ligand_code="PAR",
        )
        assert out.exists()
        content = out.read_text()
        assert "ATOM" in content or "HETATM" in content

    @requires_gemmi
    def test_cif_conversion(self, tmp_dir: Path):
        """Write a minimal mmCIF file and verify conversion completes."""
        import gemmi
        st = gemmi.Structure()
        st.name = "TEST"
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId("1", " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(1, 2, 3)
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        st.add_model(model)
        cif_path = tmp_dir / "test.cif"
        st.make_mmcif_document().write_file(str(cif_path))
        from utils.structure_io import prepare_structure
        out = prepare_structure(str(cif_path), "converted", work_dir=tmp_dir)
        assert out.exists()


# ===========================================================================
# 4. SelectionGroups
# ===========================================================================

class TestSelectionGroups:
    def test_defaults(self):
        from gromacs_analysis import SelectionGroups
        g = SelectionGroups()
        assert g.center == "1"
        assert g.output == "0"
        assert g.fit == "1"
        assert g.rmsf == "4"
        assert g.rg == "1"
        assert g.covar_fit == "4"
        assert g.covar_analysis == "4"

    def test_custom_values(self):
        from gromacs_analysis import SelectionGroups
        g = SelectionGroups(center="2", fit="5")
        assert g.center == "2"
        assert g.fit == "5"
        assert g.output == "0"   # unchanged default


# ===========================================================================
# 5. Free energy landscape (pure Python — no GROMACS)
# ===========================================================================

class TestFreeEnergyLandscape:
    @pytest.fixture
    def mock_pc_data(self) -> dict:
        rng = np.random.default_rng(42)
        return {
            "PC_1": rng.normal(0, 2, 500),
            "PC_2": rng.normal(0, 1, 500),
            "time": np.linspace(0, 100, 500),
        }

    def test_returns_required_keys(self, tmp_dir: Path, mock_pc_data: dict):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(
            md_name="md", protein_name="TEST", work_dir=str(tmp_dir)
        )
        result = sim.free_energy_landscape(pc_data=mock_pc_data)
        for key in ("G", "grad", "X", "Y", "pc1_centers", "pc2_centers",
                    "basin_pc1", "basin_pc2", "basin_G", "vmin_G", "vmax_G"):
            assert key in result, f"Missing key: {key}"

    def test_vmax_G_is_zero(self, tmp_dir: Path, mock_pc_data: dict):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        result = sim.free_energy_landscape(pc_data=mock_pc_data)
        assert result["vmax_G"] == pytest.approx(0.0)

    def test_G_shape_matches_meshgrid(self, tmp_dir: Path, mock_pc_data: dict):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        result = sim.free_energy_landscape(pc_data=mock_pc_data)
        assert result["G"].shape == result["X"].shape
        assert result["G"].shape == result["Y"].shape

    def test_basin_within_pc_range(self, tmp_dir: Path, mock_pc_data: dict):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        result = sim.free_energy_landscape(pc_data=mock_pc_data)
        assert mock_pc_data["PC_1"].min() <= result["basin_pc1"] <= mock_pc_data["PC_1"].max()
        assert mock_pc_data["PC_2"].min() <= result["basin_pc2"] <= mock_pc_data["PC_2"].max()


# ===========================================================================
# 6. Covariance → correlation (pure Python)
# ===========================================================================

class TestCovarianceToCorrelation:
    def test_diagonal_is_one(self, tmp_dir: Path):
        """The correlation of any residue with itself must be 1."""
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        # Build a proper (3N)×(3N) covariance matrix from simulated trajectory
        # so that diagonal block sums are positive (required for valid DCCM)
        n_res = 3
        rng = np.random.default_rng(0)
        n_frames = 500
        X = rng.normal(0, 1, (n_frames, n_res * 3))
        X -= X.mean(axis=0)
        cov = (X.T @ X) / n_frames   # proper positive semi-definite covariance
        flat = cov.flatten()
        dat_path = tmp_dir / "covar_TEST.dat"
        pd.DataFrame(flat).to_csv(str(dat_path), index=False, header=False)
        corr = sim.covariance_to_correlation(dat_file=str(dat_path))
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_values_bounded(self, tmp_dir: Path):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        n_res = 4
        rng = np.random.default_rng(1)
        n_frames = 500
        X = rng.normal(0, 1, (n_frames, n_res * 3))
        X -= X.mean(axis=0)
        cov = (X.T @ X) / n_frames
        flat = cov.flatten()
        dat_path = tmp_dir / "covar_TEST.dat"
        pd.DataFrame(flat).to_csv(str(dat_path), index=False, header=False)
        corr = sim.covariance_to_correlation(dat_file=str(dat_path))
        assert np.all(corr >= -1.0 - 1e-9)
        assert np.all(corr <=  1.0 + 1e-9)


# ===========================================================================
# 7. Correlation network (pure Python)
# ===========================================================================

class TestCorrelationNetwork:
    @pytest.fixture
    def mock_corr(self) -> np.ndarray:
        n = 10
        rng = np.random.default_rng(99)
        A   = rng.random((n, n))
        C   = (A + A.T) / 2
        np.fill_diagonal(C, 1.0)
        return C

    def test_returns_graph(self, tmp_dir: Path, mock_corr: np.ndarray):
        import networkx as nx
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        G = sim.correlation_network(corr=mock_corr, threshold=0.3)
        assert isinstance(G, nx.Graph)

    def test_no_self_loops(self, tmp_dir: Path, mock_corr: np.ndarray):
        import networkx as nx
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        G = sim.correlation_network(corr=mock_corr, threshold=0.3)
        assert len(list(nx.selfloop_edges(G))) == 0

    def test_threshold_reduces_edges(self, tmp_dir: Path, mock_corr: np.ndarray):
        from gromacs_analysis import GromacsAnalysis
        sim = GromacsAnalysis(md_name="md", protein_name="TEST",
                              work_dir=str(tmp_dir))
        G_low  = sim.correlation_network(corr=mock_corr, threshold=0.1)
        G_high = sim.correlation_network(corr=mock_corr, threshold=0.9)
        assert G_low.number_of_edges() >= G_high.number_of_edges()


# ===========================================================================
# 8. Config validation
# ===========================================================================

class TestApoValidation:
    def test_valid_config_passes(self):
        from sim_prep.apo import ApoSimPrepper
        sim = ApoSimPrepper(
            protein_name="prot", sim_len=10, bx_dim=1.0,
            bx_shp="dodecahedron", md_name="md", pos_ion="NA", neg_ion="CL",
        )
        sim.validate_config()   # should not raise

    def test_missing_protein_name_raises(self):
        from sim_prep.apo import ApoSimPrepper
        with pytest.raises(ValueError, match="protein_name"):
            ApoSimPrepper(sim_len=10, bx_dim=1.0, bx_shp="dodecahedron",
                          md_name="md", pos_ion="NA", neg_ion="CL",
                          protein_name="").validate_config()

    def test_sim_len_too_long_raises(self):
        from sim_prep.apo import ApoSimPrepper
        with pytest.raises(ValueError, match="sim_len"):
            ApoSimPrepper(protein_name="p", sim_len=999999, bx_dim=1.0,
                          bx_shp="dodecahedron", md_name="md",
                          pos_ion="NA", neg_ion="CL").validate_config()

    def test_invalid_box_shape_raises(self):
        from sim_prep.apo import ApoSimPrepper
        with pytest.raises(ValueError, match="bx_shp"):
            ApoSimPrepper(protein_name="p", sim_len=10, bx_dim=1.0,
                          bx_shp="sphere", md_name="md",
                          pos_ion="NA", neg_ion="CL").validate_config()

    def test_invalid_ion_raises(self):
        from sim_prep.apo import ApoSimPrepper
        with pytest.raises(ValueError, match="pos_ion"):
            ApoSimPrepper(protein_name="p", sim_len=10, bx_dim=1.0,
                          bx_shp="dodecahedron", md_name="md",
                          pos_ion="LI", neg_ion="CL").validate_config()


class TestComplexValidation:
    def test_missing_ligand_code_raises(self):
        from sim_prep.complex import CplxSimPrepper
        with pytest.raises(ValueError):
            CplxSimPrepper(protein_name="p", ligand_code="",
                           sim_len=10, bx_dim=1.0, bx_shp="dodecahedron",
                           md_name="md", pos_ion="NA", neg_ion="CL",
                           ).validate_config()

    def test_valid_complex_config_passes(self):
        from sim_prep.complex import CplxSimPrepper
        sim = CplxSimPrepper(protein_name="prot", ligand_code="LIG",
                             sim_len=10, bx_dim=1.0, bx_shp="dodecahedron",
                             md_name="md", pos_ion="NA", neg_ion="CL")
        sim.validate_config()


class TestMixMDValidation:
    def test_empty_ligands_raises(self):
        from sim_prep.mixmd import MixMDPrepper
        with pytest.raises(ValueError, match="ligands"):
            MixMDPrepper(protein_name="p", ligands=[],
                         sim_len=10, bx_dim=1.0, bx_shp="dodecahedron",
                         md_name="md", pos_ion="NA",
                         neg_ion="CL").validate_config()

    def test_ligand_missing_field_raises(self):
        from sim_prep.mixmd import MixMDPrepper
        with pytest.raises(ValueError, match="number"):
            MixMDPrepper(protein_name="p",
                         ligands=[{"code": "ACT"}],   # missing 'number'
                         sim_len=10, bx_dim=1.0, bx_shp="dodecahedron",
                         md_name="md", pos_ion="NA",
                         neg_ion="CL").validate_config()


# ===========================================================================
# 9. YAML config loader
# ===========================================================================

class TestConfigLoader:
    def test_load_apo_yaml(self, tmp_dir: Path):
        from sim_prep.config import load_config
        cfg = {
            "type": "apo", "protein_name": "test", "sim_len": 1,
            "bx_shp": "dodecahedron", "bx_dim": 1.0, "md_name": "md",
            "pos_ion": "NA", "neg_ion": "CL",
            "work_dir": str(tmp_dir),
        }
        import yaml
        p = tmp_dir / "apo.yaml"
        p.write_text(yaml.dump(cfg))
        sim = load_config(p)
        from sim_prep.apo import ApoSimPrepper
        assert isinstance(sim, ApoSimPrepper)
        assert sim.protein_name == "test"

    def test_load_json(self, tmp_dir: Path):
        from sim_prep.config import load_config
        cfg = {
            "type": "apo", "protein_name": "test_json", "sim_len": 1,
            "bx_shp": "dodecahedron", "bx_dim": 1.0, "md_name": "md",
            "pos_ion": "NA", "neg_ion": "CL",
            "work_dir": str(tmp_dir),
        }
        p = tmp_dir / "sim.json"
        p.write_text(json.dumps(cfg))
        sim = load_config(p)
        assert sim.protein_name == "test_json"

    def test_invalid_type_raises(self, tmp_dir: Path):
        from sim_prep.config import load_config
        import yaml
        cfg = {"type": "magic", "protein_name": "x"}
        p = tmp_dir / "bad.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ValueError, match="type"):
            load_config(p)

    def test_missing_file_raises(self, tmp_dir: Path):
        from sim_prep.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_dir / "ghost.yaml")

    def test_generate_apo_template(self, tmp_dir: Path):
        from sim_prep.config import generate_template
        out = tmp_dir / "apo_template.yaml"
        content = generate_template("apo", output_path=out)
        assert out.exists()
        assert "protein_name" in content
        assert "sim_len" in content

    def test_generate_complex_template(self, tmp_dir: Path):
        from sim_prep.config import generate_template
        content = generate_template("complex")
        assert "ligand_code" in content
        assert "param_ligand_name" in content

    def test_generate_invalid_type_raises(self):
        from sim_prep.config import generate_template
        with pytest.raises(ValueError):
            generate_template("quantum")


# ===========================================================================
# 10. Checkpoint system
# ===========================================================================

class TestCheckpoint:
    @pytest.fixture
    def checkpointed_sim(self, tmp_dir: Path):
        from sim_prep.checkpoint import CheckpointMixin
        from sim_prep.apo import ApoSimPrepper

        class CheckpointedApo(CheckpointMixin, ApoSimPrepper):
            pass

        sim = CheckpointedApo(
            protein_name="ckpt_test", sim_len=1, bx_dim=1.0,
            bx_shp="dodecahedron", md_name="md",
            pos_ion="NA", neg_ion="CL",
            work_dir=str(tmp_dir),
        )
        sim.validate_config()
        sim.assign_attributes()
        return sim

    def test_no_checkpoint_returns_empty(self, checkpointed_sim):
        completed = checkpointed_sim.resume_from_checkpoint()
        assert completed == []

    def test_mark_complete_persists(self, checkpointed_sim, tmp_dir: Path):
        checkpointed_sim._mark_complete("clean_pdb_file")
        # Read checkpoint file directly
        import json
        data = json.loads((tmp_dir / ".sim_checkpoint.json").read_text())
        assert "clean_pdb_file" in data["completed_steps"]

    def test_resume_loads_state(self, checkpointed_sim, tmp_dir: Path):
        checkpointed_sim._mark_complete("protein_pdb2gmx")
        checkpointed_sim._mark_complete("set_new_box")
        # Create a fresh instance and resume
        from sim_prep.checkpoint import CheckpointMixin
        from sim_prep.apo import ApoSimPrepper

        class CheckpointedApo(CheckpointMixin, ApoSimPrepper):
            pass

        sim2 = CheckpointedApo(
            protein_name="ckpt_test", sim_len=1, bx_dim=1.0,
            bx_shp="dodecahedron", md_name="md",
            pos_ion="NA", neg_ion="CL",
            work_dir=str(tmp_dir),
        )
        sim2.validate_config()
        sim2.assign_attributes()
        completed = sim2.resume_from_checkpoint()
        assert "protein_pdb2gmx" in completed
        assert "set_new_box" in completed

    def test_run_step_skips_completed(self, checkpointed_sim):
        called = []
        def dummy_step():
            called.append(True)
        checkpointed_sim._mark_complete("dummy_step")
        checkpointed_sim.run_step("dummy_step", dummy_step)
        assert called == []   # was not called

    def test_run_step_executes_new(self, checkpointed_sim):
        called = []
        def dummy_step():
            called.append(True)
        checkpointed_sim.run_step("new_step", dummy_step)
        assert called == [True]   # was called

    def test_reset_checkpoint_clears_all(self, checkpointed_sim):
        checkpointed_sim._mark_complete("step_a")
        checkpointed_sim._mark_complete("step_b")
        checkpointed_sim.reset_checkpoint()
        assert len(checkpointed_sim._completed_steps) == 0

    def test_reset_checkpoint_clears_specific(self, checkpointed_sim):
        checkpointed_sim._mark_complete("step_a")
        checkpointed_sim._mark_complete("step_b")
        checkpointed_sim.reset_checkpoint(["step_a"])
        assert "step_a" not in checkpointed_sim._completed_steps
        assert "step_b" in checkpointed_sim._completed_steps


# ===========================================================================
# 11. AmberParameteriser dependency check
# ===========================================================================

class TestAmberDependencies:
    def test_check_dependencies_returns_list(self):
        from utils.amber_params import check_dependencies
        result = check_dependencies()
        assert isinstance(result, list)

    def test_missing_tools_named(self, monkeypatch):
        """When shutil.which returns None for everything, all tools listed."""
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda x: None)
        from utils import amber_params
        # Re-run after monkeypatch
        missing = amber_params.check_dependencies()
        assert len(missing) == len(amber_params._REQUIRED_EXECUTABLES)


# ===========================================================================
# 12. Integration: GROMACS version check
# ===========================================================================

@requires_gmx
class TestGromacsIntegration:
    def test_gmx_version(self):
        import subprocess
        result = subprocess.run(
            ["gmx", "--version"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "GROMACS" in result.stdout or "GROMACS" in result.stderr
