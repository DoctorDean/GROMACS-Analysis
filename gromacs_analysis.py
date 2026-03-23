"""
gromacs_analysis.py
====================
A flexible wrapper around common GROMACS post-simulation analyses.

Workflow
--------
After a GROMACS MD simulation completes, the typical post-processing pipeline is:

    1. Remove periodic boundary conditions  →  nopbc_and_fit()
    2. Compute RMSD / RMSF / Rg            →  essential_dynamics()
    3. Covariance / PCA analysis            →  covariance_analysis()
    4. Convert covariance → DCCM            →  covariance_to_correlation()
    5. Plot the DCCM heatmap               →  plot_dccm()
    6. Build & analyse correlation network  →  correlation_network()
    7. Plot the weighted network            →  plot_correlation_network()
    8. Project trajectory onto N PCs        →  project_pca()
    9. Compute free energy landscape        →  free_energy_landscape()
   10. Plot 3D FEL + gradient magnitude     →  plot_free_energy_3d()
   11. Colour PDB by RMSF                  →  colour_pdb_by_rmsf()
   12. Colour PDB by network centrality    →  colour_pdb_by_centrality()

Example usage
-------------
    from gromacs_analysis import GromacsAnalysis, SelectionGroups

    # Minimal setup – all selection groups default to standard GROMACS indices
    sim = GromacsAnalysis(md_name="md_production", protein_name="proteinA")
    sim.nopbc_and_fit()
    sim.essential_dynamics()
    sim.covariance_analysis()

    # Custom index groups (e.g. if you have a custom .ndx file)
    groups = SelectionGroups(
        center='"Protein"',   # group used to centre the trajectory
        output="0",           # System
        fit="4",              # Backbone
        rmsf="4",             # Backbone
        rg="1",               # Protein
        covar_fit="4",        # Backbone
        covar_analysis="4",   # Backbone
    )
    sim = GromacsAnalysis(
        md_name="md_production",
        protein_name="HSP90",
        work_dir="/data/simulations/run1",
        groups=groups,
        index_file="custom.ndx",   # optional .ndx file
    )
    sim.nopbc_and_fit()
    sim.essential_dynamics()
    sim.covariance_analysis(last=10, time_unit="ns", use_pbc=False)
    sim.covariance_to_correlation()
    sim.plot_dccm(cmap="bwr")
    G = sim.correlation_network(threshold=0.3)
    sim.plot_correlation_network(G)
    pc_data = sim.project_pca(n_pcs=2)
    landscape = sim.free_energy_landscape(pc_data)
    sim.plot_free_energy_3d(landscape)
"""

import subprocess
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from Bio.PDB import PDBParser, PDBIO, Structure


# ---------------------------------------------------------------------------
# Selection group defaults
# ---------------------------------------------------------------------------

@dataclass
class SelectionGroups:
    """
    Holds the GROMACS selection group indices (or names) used across analyses.

    Defaults match the standard GROMACS group numbering for a protein-in-water
    system with no custom index file:

        0  – System
        1  – Protein
        4  – Backbone

    Override any of these if your system uses a custom .ndx file or has a
    different group layout (e.g. a membrane system, multiple chains, ligands).

    Parameters
    ----------
    center : str
        Group used to *centre* the system when removing PBC. Default ``"1"``
        (Protein).
    output : str
        Output group for trjconv calls. Default ``"0"`` (System).
    fit : str
        Group used for least-squares fitting (rot+trans). Default ``"1"``
        (Protein). Use ``"4"`` (Backbone) for a stricter fit.
    rmsf : str
        Group analysed by ``gmx rmsf``. Default ``"4"`` (Backbone).
    rg : str
        Group analysed by ``gmx gyrate``. Default ``"1"`` (Protein).
    covar_fit : str
        Least-squares fit group for ``gmx covar``. Default ``"4"``
        (Backbone).
    covar_analysis : str
        Analysis group for ``gmx covar`` (atoms whose fluctuations are
        decomposed). Default ``"4"`` (Backbone).
    """
    center: str = "1"
    output: str = "0"
    fit: str = "1"
    rmsf: str = "4"
    rg: str = "1"
    covar_fit: str = "4"
    covar_analysis: str = "4"


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class GromacsAnalysis:
    """
    Orchestrates common post-simulation GROMACS analyses.

    Parameters
    ----------
    md_name : str
        Stem of the main simulation files (without extension), e.g.
        ``"md_production"``.  The class expects ``{md_name}.tpr`` and
        ``{md_name}.xtc`` to exist in ``work_dir``.
    protein_name : str
        A short label for your protein / system, used to name output files
        (e.g. ``"HSP90"``).
    work_dir : str or Path, optional
        Directory that contains the simulation files and where outputs will
        be written.  Defaults to the current working directory.
    groups : SelectionGroups, optional
        GROMACS selection groups.  Defaults to ``SelectionGroups()`` which
        uses standard protein-in-water indices.
    index_file : str, optional
        Path to a custom ``.ndx`` index file.  When provided it is passed
        via ``-n`` to every GROMACS command that accepts one.  If ``None``
        (default) no ``-n`` flag is added and GROMACS uses its built-in
        groups.
    gmx_executable : str, optional
        Name or full path of the GROMACS executable.  Defaults to ``"gmx"``.
    """

    def __init__(
        self,
        md_name: str,
        protein_name: str,
        work_dir: Optional[str | Path] = None,
        groups: Optional[SelectionGroups] = None,
        index_file: Optional[str] = None,
        gmx_executable: str = "gmx",
    ):
        self.md_name = md_name
        self.protein_name = protein_name
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.groups = groups or SelectionGroups()
        self.index_file = index_file
        self.gmx = gmx_executable

        # Change into the working directory so all relative paths resolve
        os.chdir(self.work_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ndx_flag(self) -> list[str]:
        """Return ``['-n', index_file]`` when an index file is set, else ``[]``."""
        return ["-n", self.index_file] if self.index_file else []

    def _run(self, command: list[str], stdin: str, label: str) -> bool:
        """
        Run a GROMACS subprocess, stream stdin, and handle errors uniformly.

        Parameters
        ----------
        command : list[str]
            Full command including the ``gmx`` executable.
        stdin : str
            Newline-delimited selection string piped to the process.
        label : str
            Human-readable description printed on success/failure.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on ``CalledProcessError``.
        """
        try:
            subprocess.run(command, input=stdin, check=True, text=True)
            print(f"✓  {label}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗  {label} – command failed")
            print(f"   Command : {' '.join(command)}")
            print(f"   Error   : {e}")
            return False

    # ------------------------------------------------------------------
    # 1. Trajectory preparation
    # ------------------------------------------------------------------

    def nopbc_and_fit(self) -> None:
        """
        Prepare the raw trajectory for downstream analyses.

        Step 1 – remove periodic boundary conditions (PBC) using ``gmx
        trjconv -pbc mol -center``, centring on the group defined by
        ``groups.center`` and writing the full system.

        Step 2 – perform a least-squares rotational + translational fit
        using ``gmx trjconv -fit rot+trans`` so the protein no longer
        drifts or tumbles.

        Output files
        ------------
        ``{md_name}_noPBC.xtc``
            Trajectory with PBC removed.
        ``{md_name}_fitted.xtc``
            Trajectory additionally fitted to remove rigid-body motion.
        """
        # --- Step 1: remove PBC -------------------------------------------
        nopbc_out = f"{self.md_name}_noPBC.xtc"
        pbc_command = [
            self.gmx, "trjconv",
            "-s", f"{self.md_name}.tpr",
            "-f", f"{self.md_name}.xtc",
            "-o", nopbc_out,
            *self._ndx_flag(),
            "-pbc", "mol",
            "-center",
        ]
        # Centre on <center> group, write <output> group
        nopbc_selection = f"{self.groups.center}\n{self.groups.output}\n"
        self._run(pbc_command, nopbc_selection, "Remove periodic boundary conditions")

        # --- Step 2: fit to remove rotations / translations ----------------
        fitted_out = f"{self.md_name}_fitted.xtc"
        fit_command = [
            self.gmx, "trjconv",
            "-s", f"{self.md_name}.tpr",
            "-f", nopbc_out,
            "-o", fitted_out,
            *self._ndx_flag(),
            "-fit", "rot+trans",
        ]
        # Fit on <fit> group, write <output> group
        fit_selection = f"{self.groups.fit}\n{self.groups.output}\n"
        self._run(fit_command, fit_selection, "Fit trajectory (rot+trans)")

    # ------------------------------------------------------------------
    # 2. Essential dynamics metrics
    # ------------------------------------------------------------------

    def essential_dynamics(self, time_unit: str = "ns") -> None:
        """
        Compute the three standard metrics used to assess simulation quality
        and protein dynamics: RMSD, RMSF, and radius of gyration.

        Parameters
        ----------
        time_unit : str, optional
            Time unit for the x-axis of XVG output files.  Passed to
            ``-tu`` for commands that support it.  One of ``"fs"``,
            ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``, ``"s"``.
            Defaults to ``"ns"``.

        Output files
        ------------
        ``rmsd_{protein_name}.xvg``
            Backbone RMSD over time.
        ``rmsf_{protein_name}.xvg``
            Per-residue backbone RMSF.
        ``average_structure_{protein_name}.pdb``
            Time-averaged structure written alongside RMSF.
        ``rg_{protein_name}.xvg``
            Radius of gyration over time.
        """
        fitted_xtc = f"{self.md_name}_fitted.xtc"
        tpr = f"{self.md_name}.tpr"

        # RMSD ---------------------------------------------------------------
        rmsd_command = [
            self.gmx, "rms",
            "-s", tpr,
            "-f", fitted_xtc,
            "-o", f"rmsd_{self.protein_name}.xvg",
            *self._ndx_flag(),
            "-tu", time_unit,
        ]
        # Least-squares fit group then RMSD group (both Backbone by default)
        rmsd_selection = f"{self.groups.fit}\n{self.groups.fit}\n"
        self._run(rmsd_command, rmsd_selection, "RMSD calculation")

        # RMSF ---------------------------------------------------------------
        rmsf_command = [
            self.gmx, "rmsf",
            "-s", tpr,
            "-f", fitted_xtc,
            "-o", f"rmsf_{self.protein_name}.xvg",
            *self._ndx_flag(),
            "-res",
            "-ox", f"average_structure_{self.protein_name}.pdb",
        ]
        rmsf_selection = f"{self.groups.rmsf}\n"
        self._run(rmsf_command, rmsf_selection, "RMSF calculation")

        # Radius of gyration -------------------------------------------------
        rg_command = [
            self.gmx, "gyrate",
            "-s", tpr,
            "-f", fitted_xtc,
            "-o", f"rg_{self.protein_name}.xvg",
            *self._ndx_flag(),
            "-tu", time_unit,
        ]
        rg_selection = f"{self.groups.rg}\n"
        self._run(rg_command, rg_selection, "Radius of gyration calculation")

    # ------------------------------------------------------------------
    # 3. Covariance / PCA analysis
    # ------------------------------------------------------------------

    def covariance_analysis(
        self,
        begin: Optional[float] = None,
        end: Optional[float] = None,
        dt: Optional[float] = None,
        time_unit: str = "ns",
        fit: bool = True,
        use_pbc: bool = False,
        mass_weighted: bool = False,
        last: Optional[int] = None,
        index_file: Optional[str] = None,
    ) -> None:
        """
        Perform covariance matrix decomposition (principal component analysis)
        using ``gmx covar``.

        The covariance matrix of atomic fluctuations is built from the fitted
        trajectory, then diagonalised.  Eigenvectors are written as a
        trajectory so they can be visualised or projected onto in subsequent
        steps with ``gmx anaeig``.

        Parameters
        ----------
        begin : float, optional
            Start time for the analysis in the units set by ``time_unit``.
            Maps to ``gmx covar -b``.  Defaults to the trajectory start.
        end : float, optional
            End time for the analysis.  Maps to ``gmx covar -e``.
            Defaults to the trajectory end.
        dt : float, optional
            Only use frames separated by this time interval.  Maps to
            ``gmx covar -dt``.  Useful to reduce memory usage for long
            simulations.
        time_unit : str, optional
            Time unit used for ``-b``, ``-e``, and ``-dt``.  One of
            ``"fs"``, ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``, ``"s"``.
            Defaults to ``"ns"``.
        fit : bool, optional
            Perform a least-squares fit before building the covariance
            matrix (``-fit`` / ``-nofit``).  Defaults to ``True``.
            Set to ``False`` only if the trajectory is already fitted.
        use_pbc : bool, optional
            Take periodic boundary conditions into account
            (``-pbc`` / ``-nopbc``).  Defaults to ``False``.
        mass_weighted : bool, optional
            Use mass-weighted covariance analysis (``-mwa`` / ``-nomwa``).
            Defaults to ``False``.
        last : int, optional
            Index of the last eigenvector to write to the output trajectory.
            Maps to ``gmx covar -last``.  Defaults to all eigenvectors.
        index_file : str, optional
            Override the instance-level index file for this call only.
            If ``None`` the instance ``self.index_file`` is used.

        Output files
        ------------
        ``covar_{protein_name}.xvg``
            Eigenvalues of the covariance matrix.
        ``eigenvec_{protein_name}.trr``
            Eigenvectors written as a pseudo-trajectory.
        ``average_covar_{protein_name}.pdb``
            Average structure used as the reference for covariance.
        ``covar_{protein_name}.log``
            GROMACS log output for the covariance run.
        ``covar_{protein_name}.dat``
            Full covariance matrix in ASCII format.
        ``covar_{protein_name}.xpm``
            Covariance matrix as an XPM image (useful for quick inspection).
        """
        # Allow a one-off index file override for this call
        ndx = index_file or self.index_file
        ndx_flag = ["-n", ndx] if ndx else []

        fitted_xtc = f"{self.md_name}_fitted.xtc"
        tpr = f"{self.md_name}.tpr"

        covar_command = [
            self.gmx, "covar",
            "-s", tpr,
            "-f", fitted_xtc,
            *ndx_flag,
            # --- outputs ---
            "-o",     f"covar_{self.protein_name}.xvg",
            "-v",     f"eigenvec_{self.protein_name}.trr",
            "-av",    f"average_covar_{self.protein_name}.pdb",
            "-l",     f"covar_{self.protein_name}.log",
            "-ascii", f"covar_{self.protein_name}.dat",
            "-xpm",   f"covar_{self.protein_name}.xpm",
            # --- time unit ---
            "-tu", time_unit,
            # --- boolean flags ---
            "-fit"   if fit         else "-nofit",
            "-pbc"   if use_pbc     else "-nopbc",
            "-mwa"   if mass_weighted else "-nomwa",
        ]

        # Optional time window and stride
        if begin is not None:
            covar_command += ["-b", str(begin)]
        if end is not None:
            covar_command += ["-e", str(end)]
        if dt is not None:
            covar_command += ["-dt", str(dt)]
        if last is not None:
            covar_command += ["-last", str(last)]

        # First selection: group for least-squares fit
        # Second selection: group whose atomic fluctuations are decomposed
        covar_selection = (
            f"{self.groups.covar_fit}\n"
            f"{self.groups.covar_analysis}\n"
        )

        self._run(
            covar_command,
            covar_selection,
            f"Covariance / PCA analysis (fit={'yes' if fit else 'no'}, "
            f"last={last or 'all'} eigenvectors)",
        )

    # ------------------------------------------------------------------
    # 4. Covariance → dynamic cross-correlation matrix
    # ------------------------------------------------------------------

    def covariance_to_correlation(
        self,
        dat_file: Optional[str] = None,
    ) -> np.ndarray:
        """
        Parse the ASCII covariance matrix produced by ``gmx covar -ascii``
        and convert it to a per-residue dynamic cross-correlation matrix
        (DCCM).

        The DCCM entry C(i,j) is defined as:

        .. code-block:: text

            C(i,j) = cov(i,j) / sqrt( cov(i,i) * cov(j,j) )

        Values range from -1 (perfectly anti-correlated) to +1 (perfectly
        correlated).

        The raw ``gmx covar -ascii`` file contains one value per line
        representing the flattened (3N × 3N) covariance matrix, where N is
        the number of selected atoms / residues.  This method sums the 3×3
        blocks along each atom pair to produce an N×N residue-level matrix
        before normalising.

        Parameters
        ----------
        dat_file : str, optional
            Path to the ASCII covariance matrix file.  Defaults to
            ``covar_{protein_name}.dat`` in ``work_dir``.

        Output files
        ------------
        ``cross_corr_{protein_name}.csv``
            Space-delimited DCCM saved for downstream use.

        Returns
        -------
        np.ndarray
            (N, N) cross-correlation matrix.
        """
        dat_path = dat_file or str(self.work_dir / f"covar_{self.protein_name}.dat")

        # Read the flat covariance values produced by gmx covar -ascii
        covar = pd.read_csv(dat_path, header=None, names=["value"])

        # Number of residues: total elements = (3N)^2  →  N = sqrt(len / 9)
        resnum = int(math.sqrt(len(covar.index) / 3))

        # ---- Rebuild the N×N residue-level covariance matrix ---------------
        # The 3N×3N matrix is stored row-major.  For each residue pair (i, j)
        # the relevant 3×3 block is summed to get a scalar covariance.
        all_results = pd.DataFrame()

        for i in range(resnum):
            three_step = pd.DataFrame()
            for j in range((i * resnum) * 3, int(len(covar) / resnum) * (i + 1), resnum):
                block = covar.iloc[j : resnum + j].reset_index(drop=True)
                three_step = pd.concat([three_step, block], ignore_index=True, axis=1)
            all_results = pd.concat([all_results, three_step], ignore_index=True, axis=0)

        all_results["sum"] = all_results.sum(axis=1)
        cov_matrix = all_results["sum"].to_numpy().reshape(resnum, resnum)

        # ---- Normalise to cross-correlation --------------------------------
        corr = np.zeros((resnum, resnum))
        for i in range(resnum):
            for j in range(resnum):
                denom = math.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                corr[i, j] = cov_matrix[i, j] / denom if denom != 0 else 0.0

        # ---- Persist -------------------------------------------------------
        out_path = self.work_dir / f"cross_corr_{self.protein_name}.csv"
        np.savetxt(str(out_path), corr, delimiter=" ", fmt="%s")
        print(f"✓  DCCM saved → {out_path}")

        return corr

    # ------------------------------------------------------------------
    # 5. Plot DCCM
    # ------------------------------------------------------------------

    def plot_dccm(
        self,
        corr: Optional[np.ndarray] = None,
        csv_file: Optional[str] = None,
        cmap: str = "bwr",
        vmin: float = -1.0,
        vmax: float = 1.0,
        major_tick_interval: int = 100,
        minor_tick_interval: int = 50,
        font: str = "Times New Roman",
        title_fontsize: int = 28,
        tick_fontsize: int = 18,
        colorbar_fontsize: int = 18,
        dpi: int = 500,
        figsize: tuple[int, int] = (11, 9),
    ) -> None:
        """
        Plot the dynamic cross-correlation matrix (DCCM) as a heatmap and
        save it as a JPEG.

        Parameters
        ----------
        corr : np.ndarray, optional
            Pre-computed (N, N) correlation matrix.  If ``None``, the saved
            ``cross_corr_{protein_name}.csv`` is loaded automatically.
        csv_file : str, optional
            Explicit path to a CSV file to load when ``corr`` is ``None``.
            Falls back to ``cross_corr_{protein_name}.csv``.
        cmap : str, optional
            Matplotlib colormap name.  Good choices for diverging data:
            ``"bwr"``, ``"seismic"``, ``"RdBu"``, ``"coolwarm"``,
            ``"PiYG"``, ``"viridis"``.  Defaults to ``"bwr"``.
        vmin, vmax : float, optional
            Colour scale limits.  Defaults to [-1, 1].
        major_tick_interval : int, optional
            Spacing of labelled (major) axis ticks in residue units.
            Defaults to 100.
        minor_tick_interval : int, optional
            Spacing of unlabelled (minor) axis ticks.  Defaults to 50.
        font : str, optional
            Font family for all text elements.  Defaults to
            ``"Times New Roman"``.
        title_fontsize : int, optional
            Font size of the plot title.  Defaults to 28.
        tick_fontsize : int, optional
            Font size of axis tick labels.  Defaults to 18.
        colorbar_fontsize : int, optional
            Font size of colour-bar tick labels.  Defaults to 18.
        dpi : int, optional
            Resolution of the saved figure in dots per inch.  Defaults to
            500.
        figsize : tuple of int, optional
            Figure dimensions ``(width, height)`` in inches.  Defaults to
            ``(11, 9)``.

        Output files
        ------------
        ``cross_corr_{protein_name}.jpeg``
            Heatmap of the DCCM.
        """
        # Load matrix if not passed in directly
        if corr is None:
            src = csv_file or str(self.work_dir / f"cross_corr_{self.protein_name}.csv")
            corr = np.loadtxt(src)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, origin="lower")

        # Colour bar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname(font)

        # Title and axis ticks
        ax.set_title(
            f"Dynamic Cross-Correlation – {self.protein_name}",
            fontname=font,
            fontsize=title_fontsize,
            pad=15,
        )
        ax.tick_params(labelsize=tick_fontsize)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(font)

        ax.set_xticks(np.arange(0, corr.shape[1], major_tick_interval))
        ax.set_yticks(np.arange(0, corr.shape[0], major_tick_interval))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
        ax.tick_params(which="minor", direction="out", length=4, color="gray")

        fig.tight_layout()

        out_path = self.work_dir / f"cross_corr_{self.protein_name}.jpeg"
        fig.savefig(str(out_path), dpi=dpi)
        plt.close(fig)
        print(f"✓  DCCM figure saved → {out_path}")

    # ------------------------------------------------------------------
    # 6. Correlation network
    # ------------------------------------------------------------------

    def correlation_network(
        self,
        corr: Optional[np.ndarray] = None,
        csv_file: Optional[str] = None,
        threshold: float = 0.3,
    ) -> nx.Graph:
        """
        Build a weighted, undirected correlation network from the DCCM and
        print a comprehensive set of graph-theoretic metrics.

        Edges are added between residue pairs (i, j) whenever
        ``|C(i,j)| > threshold``.  Self-loops are removed before any metric
        is computed.

        Parameters
        ----------
        corr : np.ndarray, optional
            Pre-computed (N, N) correlation matrix.  If ``None``, the saved
            CSV is loaded.
        csv_file : str, optional
            Explicit path to the CSV file used when ``corr`` is ``None``.
        threshold : float, optional
            Absolute correlation threshold for adding an edge.  Increase to
            focus on strongly correlated pairs; decrease to capture weaker
            allosteric communication.  Defaults to ``0.3``.

        Returns
        -------
        networkx.Graph
            The weighted correlation graph, with self-loops removed, for use
            in downstream visualisation or further analysis.

        Printed metrics
        ---------------
        Connectivity, average path length, average clustering coefficient,
        max degree / betweenness / closeness / eigenvector centrality, max
        edge betweenness, degree assortativity, k-core size, and articulation
        points.
        """
        # Load matrix if not provided
        if corr is None:
            src = csv_file or str(self.work_dir / f"cross_corr_{self.protein_name}.csv")
            corr = np.loadtxt(src).astype(float)

        G = nx.Graph()
        n = len(corr)
        for i in range(n):
            for j in range(n):
                if abs(corr[i, j]) > threshold:
                    G.add_edge(i, j, weight=float(corr[i, j]))

        # Remove self-loops before all metric calculations
        G.remove_edges_from(nx.selfloop_edges(G))

        # ---- Metrics -------------------------------------------------------
        is_connected = nx.is_connected(G)

        degree_centrality  = nx.degree_centrality(G)
        betweenness        = nx.betweenness_centrality(G)
        closeness          = nx.closeness_centrality(G)
        eigenvector        = nx.eigenvector_centrality(G, max_iter=1000)
        edge_betweenness   = nx.edge_betweenness_centrality(G)
        assortativity      = nx.degree_assortativity_coefficient(G)
        clustering         = nx.average_clustering(G)
        core               = nx.k_core(G)
        articulation_pts   = list(nx.articulation_points(G))

        try:
            avg_path_length = nx.average_shortest_path_length(G) if is_connected else None
        except Exception:
            avg_path_length = None

        # ---- Print ---------------------------------------------------------
        sep = "-" * 50
        print(f"\n{sep}")
        print(f"  Network metrics – {self.protein_name}  (threshold={threshold})")
        print(sep)
        print(f"  Nodes / Edges        : {G.number_of_nodes()} / {G.number_of_edges()}")
        print(f"  Connected            : {is_connected}")
        print(f"  Avg path length      : "
              f"{avg_path_length:.4f}" if avg_path_length else "  Avg path length      : graph not fully connected")
        print(f"  Avg clustering coeff : {clustering:.4f}")
        print(f"  Max degree centrality      : {max(degree_centrality.values()):.4f}")
        print(f"  Max betweenness centrality : {max(betweenness.values()):.4f}")
        print(f"  Max closeness centrality   : {max(closeness.values()):.4f}")
        print(f"  Max eigenvector centrality : {max(eigenvector.values()):.4f}")
        print(f"  Max edge betweenness       : {max(edge_betweenness.values()):.4f}")
        print(f"  Degree assortativity       : {assortativity:.4f}")
        print(f"  k-core size                : {len(core)}")
        print(f"  Articulation points ({len(articulation_pts)})  : {articulation_pts}")
        print(sep + "\n")

        return G

    # ------------------------------------------------------------------
    # 7. Plot weighted correlation network
    # ------------------------------------------------------------------

    def plot_correlation_network(
        self,
        G: nx.Graph,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
        node_size: int = 300,
        font_size: int = 6,
        edge_scale: float = 2.0,
        layout_seed: int = 42,
        node_cmap: str = "viridis",
        edge_cmap: str = "Greys",
        dpi: int = 300,
    ) -> None:
        """
        Draw the weighted correlation network and save it as a PNG.

        Node colour encodes degree centrality (darker = more connected).
        Edge colour and width encode the absolute correlation weight.
        Self-loops are excluded from the drawing.

        Parameters
        ----------
        G : networkx.Graph
            Graph returned by :meth:`correlation_network`.
        title : str, optional
            Figure title.  Defaults to
            ``"Weighted Correlation Network – {protein_name}"``.
        figsize : tuple of int, optional
            Figure dimensions ``(width, height)`` in inches.  Defaults to
            ``(10, 8)``.
        node_size : int, optional
            Size of each node marker.  Defaults to 300.
        font_size : int, optional
            Font size of node labels.  Defaults to 6.
        edge_scale : float, optional
            Multiplier applied to edge weights to set line width.  Increase
            for thicker edges.  Defaults to 2.0.
        layout_seed : int, optional
            Random seed for the spring layout algorithm, ensuring
            reproducible figures.  Defaults to 42.
        node_cmap : str, optional
            Colormap for node colours.  Defaults to ``"viridis"``.
        edge_cmap : str, optional
            Colormap for edge colours.  Defaults to ``"Greys"``.
        dpi : int, optional
            Resolution of the saved figure.  Defaults to 300.

        Output files
        ------------
        ``network_{protein_name}.png``
            Weighted network figure.
        """
        title = title or f"Weighted Correlation Network – {self.protein_name}"

        pos     = nx.spring_layout(G, seed=layout_seed)
        edges   = [(u, v) for u, v, _ in G.edges(data=True) if u != v]
        weights = [abs(d["weight"]) for u, v, d in G.edges(data=True) if u != v]

        centrality   = nx.degree_centrality(G)
        node_colors  = [centrality[node] for node in G.nodes()]

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=node_size,
            node_color=node_colors,
            cmap=plt.get_cmap(node_cmap),
            edgelist=edges,
            edge_color=weights,
            edge_cmap=plt.get_cmap(edge_cmap),
            width=[w * edge_scale for w in weights],
            linewidths=0.5,
            font_size=font_size,
        )
        ax.set_title(title)
        fig.tight_layout()

        out_path = self.work_dir / f"network_{self.protein_name}.png"
        fig.savefig(str(out_path), dpi=dpi)
        plt.close(fig)
        print(f"✓  Network figure saved → {out_path}")

    # ------------------------------------------------------------------
    # 8. Project trajectory onto principal components
    # ------------------------------------------------------------------

    def project_pca(
        self,
        first: int = 1,
        last: int = 2,
        begin: Optional[float] = None,
        end: Optional[float] = None,
        dt: Optional[float] = None,
        skip: int = 1,
        time_unit: str = "ns",
        eigenvec_file: Optional[str] = None,
        index_file: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """
        Project the fitted trajectory onto a set of principal components
        using ``gmx anaeig``, then parse the resulting XVG files into
        NumPy arrays ready for free-energy landscape calculation.

        ``gmx covar`` must have been run first so that the eigenvector
        file (``eigenvec_{protein_name}.trr``) exists.

        By default this projects onto PC1 and PC2 (``first=1``, ``last=2``).
        Increase ``last`` to capture more PCs, e.g. ``last=3`` for a 3-D
        landscape.

        Parameters
        ----------
        first : int, optional
            Index of the first eigenvector to project onto.  Defaults to
            ``1`` (PC1).
        last : int, optional
            Index of the last eigenvector to project onto.  Defaults to
            ``2`` (PC2).  Setting ``last=N`` produces N projections.
        begin : float, optional
            Start time for the analysis (in ``time_unit``).  Defaults to
            the trajectory start.
        end : float, optional
            End time for the analysis.  Defaults to the trajectory end.
        dt : float, optional
            Only use frames separated by this time interval.
        skip : int, optional
            Use every ``skip``-th frame.  Maps to ``gmx anaeig -skip``.
            Defaults to 1 (every frame).
        time_unit : str, optional
            Time unit for ``-b`` / ``-e`` / ``-dt`` and XVG output.
            Defaults to ``"ns"``.
        eigenvec_file : str, optional
            Path to the eigenvector TRR produced by ``gmx covar``.
            Defaults to ``eigenvec_{protein_name}.trr``.
        index_file : str, optional
            Override the instance-level index file for this call only.

        Output files
        ------------
        ``proj_PC{i}_{protein_name}.xvg``  (one per PC)
            Time series of the projection of each frame onto eigenvector *i*.
        ``pc_projections_{protein_name}.csv``
            Tidy CSV with columns ``time``, ``PC_1``, ``PC_2``, …
            suitable for direct use in :meth:`free_energy_landscape`.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping ``"PC_1"``, ``"PC_2"``, … to 1-D arrays of
            projection values (length = number of trajectory frames used).
            Also includes a ``"time"`` key.
        """
        eigenvec = eigenvec_file or str(self.work_dir / f"eigenvec_{self.protein_name}.trr")
        ndx = index_file or self.index_file
        ndx_flag = ["-n", ndx] if ndx else []
        fitted_xtc = f"{self.md_name}_fitted.xtc"
        tpr = f"{self.md_name}.tpr"

        pc_arrays: dict[str, np.ndarray] = {}

        for pc_idx in range(first, last + 1):
            proj_out = str(self.work_dir / f"proj_PC{pc_idx}_{self.protein_name}.xvg")

            anaeig_command = [
                self.gmx, "anaeig",
                "-v",    eigenvec,
                "-f",    fitted_xtc,
                "-s",    tpr,
                *ndx_flag,
                "-proj", proj_out,
                "-first", str(pc_idx),
                "-last",  str(pc_idx),
                "-tu",    time_unit,
                "-skip",  str(skip),
            ]

            if begin is not None:
                anaeig_command += ["-b", str(begin)]
            if end is not None:
                anaeig_command += ["-e", str(end)]
            if dt is not None:
                anaeig_command += ["-dt", str(dt)]

            ok = self._run(
                anaeig_command,
                stdin="",   # anaeig reads groups from the eigenvec file header
                label=f"Project trajectory onto PC{pc_idx}",
            )

            if ok:
                time_vals, proj_vals = self._parse_xvg(proj_out)
                if "time" not in pc_arrays:
                    pc_arrays["time"] = time_vals
                pc_arrays[f"PC_{pc_idx}"] = proj_vals

        # ---- Save tidy CSV -------------------------------------------------
        if pc_arrays:
            df = pd.DataFrame(pc_arrays)
            csv_path = self.work_dir / f"pc_projections_{self.protein_name}.csv"
            df.to_csv(str(csv_path), index=False)
            print(f"✓  PC projections saved → {csv_path}")

        return pc_arrays

    # ------------------------------------------------------------------
    # Internal: XVG parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xvg(path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse a two-column GROMACS XVG file, skipping comment / label lines
        that begin with ``#`` or ``@``.

        Parameters
        ----------
        path : str
            Path to the XVG file.

        Returns
        -------
        tuple of np.ndarray
            ``(x_values, y_values)`` as float arrays.
        """
        x, y = [], []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "@")):
                    continue
                parts = line.split()
                x.append(float(parts[0]))
                y.append(float(parts[1]))
        return np.array(x), np.array(y)

    # ------------------------------------------------------------------
    # 9. Free energy landscape
    # ------------------------------------------------------------------

    def free_energy_landscape(
        self,
        pc_data: Optional[dict[str, np.ndarray]] = None,
        csv_file: Optional[str] = None,
        pc_x: str = "PC_1",
        pc_y: str = "PC_2",
        bin_width: float = 2.0,
        sigma: float = 1.5,
        temperature: float = 300.0,
        kB: float = 0.008314,
    ) -> dict:
        """
        Compute a 2-D free energy landscape (FEL) from PC projections using
        the relationship:

        .. code-block:: text

            G(PC1, PC2) = -kB·T · ln P(PC1, PC2)

        The landscape is normalised so that the global maximum is 0 (basins
        appear as negative wells), smoothed with a Gaussian filter, and
        renormalised again after smoothing.

        The gradient magnitude ``|∇G|`` is also computed, which highlights
        transition state regions and barriers between basins.

        Parameters
        ----------
        pc_data : dict[str, np.ndarray], optional
            Dictionary as returned by :meth:`project_pca`, containing at
            least keys ``pc_x`` and ``pc_y``.  If ``None``, the saved CSV
            is loaded.
        csv_file : str, optional
            Explicit path to the CSV.  Falls back to
            ``pc_projections_{protein_name}.csv``.
        pc_x : str, optional
            Column name to use as the x-axis PC.  Defaults to ``"PC_1"``.
        pc_y : str, optional
            Column name to use as the y-axis PC.  Defaults to ``"PC_2"``.
        bin_width : float, optional
            Width of each histogram bin in PC-space units (same units as
            the projection XVG, typically nm or Å).  Smaller values give
            finer resolution at the cost of noisier estimates.  Defaults to
            ``2.0``.
        sigma : float, optional
            Standard-deviation of the Gaussian smoothing kernel applied to
            the free energy surface.  Values in the range 1–3 are typical;
            increase for smoother surfaces.  Defaults to ``1.5``.
        temperature : float, optional
            Simulation temperature in Kelvin, used to scale the free energy.
            Defaults to ``300.0``.
        kB : float, optional
            Boltzmann constant in units consistent with the desired energy
            output.  Defaults to ``0.008314`` kJ·mol⁻¹·K⁻¹.

        Returns
        -------
        dict
            A results dictionary with the following keys:

            ``"G"``         – (M, N) smoothed free energy array (kJ/mol)
            ``"grad"``      – (M, N) gradient magnitude array
            ``"X"`` / ``"Y"``  – meshgrid arrays of bin centres
            ``"pc1_centers"`` / ``"pc2_centers"`` – 1-D bin centre arrays
            ``"xedges"`` / ``"yedges"`` – histogram bin edges
            ``"basin_pc1"`` / ``"basin_pc2"`` / ``"basin_G"``
                               – coordinates and energy of the global minimum
            ``"vmin_G"`` / ``"vmax_G"``  – colour scale limits for G
            ``"vmin_grad"`` / ``"vmax_grad"`` – colour scale limits for gradient
            ``"pc_x"`` / ``"pc_y"``  – axis labels used
        """
        # ---- Load data -----------------------------------------------------
        if pc_data is None:
            src = csv_file or str(self.work_dir / f"pc_projections_{self.protein_name}.csv")
            df = pd.read_csv(src)
            pc1_vals = df[pc_x].values
            pc2_vals = df[pc_y].values
        else:
            pc1_vals = pc_data[pc_x]
            pc2_vals = pc_data[pc_y]

        # ---- Histogram & probability ----------------------------------------
        pc1_bins = np.arange(pc1_vals.min(), pc1_vals.max() + bin_width, bin_width)
        pc2_bins = np.arange(pc2_vals.min(), pc2_vals.max() + bin_width, bin_width)

        H, xedges, yedges = np.histogram2d(pc1_vals, pc2_vals,
                                            bins=[pc1_bins, pc2_bins])
        H = H.T  # rows = PC2, cols = PC1 (imshow convention)

        epsilon = 1e-10
        P = H / np.sum(H)
        P[P == 0] = epsilon

        # ---- Free energy ---------------------------------------------------
        G = -kB * temperature * np.log(P)
        G -= np.max(G)                             # shift: max → 0
        G = gaussian_filter(G, sigma=sigma)        # smooth
        G -= np.max(G)                             # re-shift after smoothing

        # ---- Gradient magnitude --------------------------------------------
        pc1_centers = 0.5 * (xedges[:-1] + xedges[1:])
        pc2_centers = 0.5 * (yedges[:-1] + yedges[1:])
        dpc1 = pc1_centers[1] - pc1_centers[0]
        dpc2 = pc2_centers[1] - pc2_centers[0]

        dG_dy, dG_dx = np.gradient(G, dpc2, dpc1)  # row = PC2, col = PC1
        grad = np.sqrt(dG_dx**2 + dG_dy**2)

        X, Y = np.meshgrid(pc1_centers, pc2_centers)

        # ---- Basin location ------------------------------------------------
        idx = np.unravel_index(np.argmin(G), G.shape)
        basin_pc1 = pc1_centers[idx[1]]
        basin_pc2 = pc2_centers[idx[0]]
        basin_G   = float(G[idx])

        print(
            f"✓  FEL computed  |  basin: ({basin_pc1:.2f}, {basin_pc2:.2f}) "
            f"G = {basin_G:.2f} kJ/mol  |  bins: {H.shape[1]}×{H.shape[0]}"
        )

        return {
            "G":           G,
            "grad":        grad,
            "X":           X,
            "Y":           Y,
            "pc1_centers": pc1_centers,
            "pc2_centers": pc2_centers,
            "xedges":      xedges,
            "yedges":      yedges,
            "basin_pc1":   basin_pc1,
            "basin_pc2":   basin_pc2,
            "basin_G":     basin_G,
            "vmin_G":      float(np.min(G)),
            "vmax_G":      0.0,
            "vmin_grad":   0.0,
            "vmax_grad":   float(np.max(grad)),
            "pc_x":        pc_x,
            "pc_y":        pc_y,
        }

    # ------------------------------------------------------------------
    # 10. Plot 3-D FEL + gradient magnitude
    # ------------------------------------------------------------------

    def plot_free_energy_3d(
        self,
        landscape: dict,
        cmap_G: str = "viridis",
        cmap_grad: str = "plasma",
        elev: float = 30.0,
        azim: float = -60.0,
        figsize_3d: tuple[int, int] = (16, 7),
        figsize_grad: tuple[int, int] = (16, 12),
        alpha: float = 0.92,
        n_contour_levels: int = 10,
        dpi: int = 150,
    ) -> None:
        """
        Produce two publication-quality figures from a free energy landscape
        dictionary returned by :meth:`free_energy_landscape`.

        **Figure 1** (``fel_3d_{protein_name}.png``) — a single 3-D surface
        of ΔG coloured by energy depth.  The global free energy minimum is
        marked with a red scatter point and labelled with its energy value.

        **Figure 2** (``fel_gradient_{protein_name}.png``) — a 2×2 panel:

        * Top row: 3-D gradient-magnitude surfaces for each system.
        * Bottom row: 2-D heatmaps of |∇G| with free energy contour lines
          overlaid, so that both the barrier heights and the basin geometry
          are visible simultaneously.

        Parameters
        ----------
        landscape : dict
            Output of :meth:`free_energy_landscape`.  Pass a single
            landscape or see notes below for comparing two systems.
        cmap_G : str, optional
            Colormap for the free energy surface.  Defaults to
            ``"viridis"``.
        cmap_grad : str, optional
            Colormap for the gradient magnitude panels.  Defaults to
            ``"plasma"``.
        elev : float, optional
            Elevation angle (degrees) for the 3-D view.  Defaults to 30.
        azim : float, optional
            Azimuth angle (degrees) for the 3-D view.  Defaults to -60.
        figsize_3d : tuple of int, optional
            Figure size for the 3-D FEL plot.  Defaults to ``(16, 7)``.
        figsize_grad : tuple of int, optional
            Figure size for the gradient magnitude figure.  Defaults to
            ``(16, 12)``.
        alpha : float, optional
            Surface transparency (0 = transparent, 1 = opaque).  Defaults
            to 0.92.
        n_contour_levels : int, optional
            Number of contour levels overlaid on the 2-D gradient heatmaps.
            Defaults to 10.
        dpi : int, optional
            Resolution of the saved figures.  Defaults to 150.

        Notes
        -----
        To compare two simulations side-by-side (the original intended use
        case for this code), call :meth:`free_energy_landscape` twice and
        then call this method for each result separately.  Alternatively,
        use the returned figure objects to compose a custom multi-panel
        layout.

        Output files
        ------------
        ``fel_3d_{protein_name}.png``
            3-D free energy surface.
        ``fel_gradient_{protein_name}.png``
            Gradient magnitude (3-D + 2-D heatmap with contours).
        """
        G    = landscape["G"]
        grad = landscape["grad"]
        X, Y = landscape["X"], landscape["Y"]
        pc1c = landscape["pc1_centers"]
        pc2c = landscape["pc2_centers"]
        bx   = landscape["basin_pc1"]
        by   = landscape["basin_pc2"]
        bz   = landscape["basin_G"]
        vmin_G    = landscape["vmin_G"]
        vmin_grad = landscape["vmin_grad"]
        vmax_grad = landscape["vmax_grad"]
        pc_x = landscape["pc_x"]
        pc_y = landscape["pc_y"]

        # ============================================================
        # Figure 1 – 3-D free energy surface
        # ============================================================
        fig1, ax1 = plt.subplots(
            1, 1, figsize=figsize_3d,
            subplot_kw={"projection": "3d"},
        )

        surf = ax1.plot_surface(
            X, Y, G,
            cmap=plt.get_cmap(cmap_G),
            vmin=vmin_G, vmax=0.0,
            linewidth=0, antialiased=True, alpha=alpha,
        )
        ax1.set_title(
            f"Free Energy Landscape – {self.protein_name}",
            fontsize=13, pad=12,
        )
        ax1.set_xlabel(pc_x, labelpad=8)
        ax1.set_ylabel(pc_y, labelpad=8)
        ax1.set_zlabel("ΔG (kJ/mol)", labelpad=8)
        ax1.set_zlim(vmin_G, 0)
        ax1.view_init(elev=elev, azim=azim)

        # Mark deepest basin
        ax1.scatter([bx], [by], [bz], color="red", s=60, zorder=5)
        ax1.text(
            bx, by, bz - abs(vmin_G) * 0.05,
            f"{bz:.1f} kJ/mol",
            color="red", fontsize=9, fontweight="bold", ha="center",
        )

        cbar1 = fig1.colorbar(surf, ax=ax1, shrink=0.55, pad=0.05, aspect=20)
        cbar1.set_label("ΔG (kJ/mol)", fontsize=11)

        fig1.tight_layout()
        out1 = self.work_dir / f"fel_3d_{self.protein_name}.png"
        fig1.savefig(str(out1), dpi=dpi, bbox_inches="tight")
        plt.close(fig1)
        print(f"✓  3-D FEL figure saved → {out1}")

        # ============================================================
        # Figure 2 – Gradient magnitude (3-D surface + 2-D heatmap)
        # ============================================================
        fig2 = plt.figure(figsize=figsize_grad)
        extent = [pc1c.min(), pc1c.max(), pc2c.min(), pc2c.max()]

        # --- 3-D gradient surface ---
        ax3d = fig2.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot_surface(
            X, Y, grad,
            cmap=plt.get_cmap(cmap_grad),
            vmin=vmin_grad, vmax=vmax_grad,
            linewidth=0, antialiased=True, alpha=alpha,
        )
        ax3d.set_title(
            f"{self.protein_name}\nGradient Magnitude (3-D)",
            fontsize=11, pad=10,
        )
        ax3d.set_xlabel(pc_x, labelpad=6)
        ax3d.set_ylabel(pc_y, labelpad=6)
        ax3d.set_zlabel("|∇G| (kJ/mol per PC unit)", labelpad=6)
        ax3d.set_zlim(vmin_grad, vmax_grad)
        ax3d.view_init(elev=elev, azim=azim)

        # Project basin position onto gradient surface
        grad_at_basin = grad[
            np.argmin(np.abs(pc2c - by)),
            np.argmin(np.abs(pc1c - bx)),
        ]
        ax3d.scatter(
            [bx], [by], [grad_at_basin],
            color="cyan", s=50, zorder=5, label="Basin minimum",
        )

        # --- 2-D heatmap + free energy contours ---
        ax2d = fig2.add_subplot(1, 2, 2)
        im = ax2d.imshow(
            grad,
            origin="lower", extent=extent,
            aspect="auto",
            cmap=plt.get_cmap(cmap_grad),
            vmin=vmin_grad, vmax=vmax_grad,
        )
        contour_levels = np.linspace(vmin_G * 0.9, 0, n_contour_levels)
        cs = ax2d.contour(
            X, Y, G,
            levels=contour_levels,
            colors="white", linewidths=0.7, alpha=0.6,
        )
        ax2d.clabel(cs, fmt="%.1f", fontsize=7, colors="white")
        ax2d.scatter([bx], [by], color="cyan", s=60,
                     zorder=5, label="Basin minimum")
        ax2d.legend(fontsize=8, loc="upper right")
        ax2d.set_title(
            f"{self.protein_name}\nGradient Magnitude (2-D) + ΔG contours",
            fontsize=11,
        )
        ax2d.set_xlabel(pc_x)
        ax2d.set_ylabel(pc_y)

        cbar2 = fig2.colorbar(im, ax=fig2.axes, shrink=0.6, pad=0.04, aspect=25)
        cbar2.set_label("|∇G| (kJ/mol per PC unit)", fontsize=11)
        fig2.suptitle(
            f"Gradient Magnitude of Free Energy Landscape – {self.protein_name}",
            fontsize=14,
        )

        out2 = self.work_dir / f"fel_gradient_{self.protein_name}.png"
        fig2.savefig(str(out2), dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        print(f"✓  Gradient figure saved → {out2}")

    # ------------------------------------------------------------------
    # Internal: apply a per-residue value dict to a Bio.PDB structure
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_bfactors(
        structure,
        residue_values: dict[int, float],
        default: float = 0.0,
    ) -> None:
        """
        Write per-residue scalar values into the B-factor field of every
        atom in a Bio.PDB ``Structure`` object, in-place.

        Residues are matched by their sequence number (``residue.id[1]``).
        Residues with no entry in ``residue_values`` receive ``default``.

        Parameters
        ----------
        structure : Bio.PDB.Structure
            Parsed structure object (modified in-place).
        residue_values : dict[int, float]
            Mapping of ``{residue_sequence_number: value}``.
        default : float, optional
            B-factor written for residues absent from ``residue_values``.
            Defaults to ``0.0``.
        """
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.id[1]
                    value = residue_values.get(res_id, default)
                    for atom in residue:
                        atom.bfactor = value

    @staticmethod
    def _save_pdb(structure, out_path: Path) -> None:
        """Write a Bio.PDB structure to *out_path* using ``PDBIO``."""
        io = PDBIO()
        io.set_structure(structure)
        with out_path.open("w") as fh:
            io.save(fh)

    # ------------------------------------------------------------------
    # 11. Colour PDB by RMSF
    # ------------------------------------------------------------------

    def colour_pdb_by_rmsf(
        self,
        pdb_file: Optional[str] = None,
        xvg_file: Optional[str] = None,
        output_file: Optional[str] = None,
        default_bfactor: float = 0.0,
    ) -> Path:
        """
        Replace the B-factor column of a PDB file with per-residue RMSF
        values read from a GROMACS XVG file and write a new PDB.

        The output can be opened directly in PyMOL or ChimeraX and coloured
        by B-factor to produce a per-residue flexibility map.  In PyMOL::

            spectrum b, blue_white_red, minimum=0, maximum=<max_rmsf>

        Parameters
        ----------
        pdb_file : str, optional
            Path to the input PDB.  Defaults to
            ``average_structure_{protein_name}.pdb`` (the average structure
            written by ``gmx rmsf -ox``).
        xvg_file : str, optional
            Path to the RMSF XVG file produced by ``gmx rmsf``.  Defaults
            to ``rmsf_{protein_name}.xvg``.
        output_file : str, optional
            Path for the output PDB.  Defaults to
            ``bfactor_rmsf_{protein_name}.pdb`` in ``work_dir``.
        default_bfactor : float, optional
            B-factor assigned to residues not present in the XVG (e.g.
            terminal residues skipped by GROMACS).  Defaults to ``0.0``.

        Returns
        -------
        Path
            Path to the written PDB file.
        """
        pdb_path = Path(pdb_file) if pdb_file else (
            self.work_dir / f"average_structure_{self.protein_name}.pdb"
        )
        xvg_path = Path(xvg_file) if xvg_file else (
            self.work_dir / f"rmsf_{self.protein_name}.xvg"
        )
        out_path = Path(output_file) if output_file else (
            self.work_dir / f"bfactor_rmsf_{self.protein_name}.pdb"
        )

        # Parse RMSF XVG → {residue_number: rmsf_value}
        res_nums, rmsf_vals = self._parse_xvg(str(xvg_path))
        residue_values = {int(r): float(v) for r, v in zip(res_nums, rmsf_vals)}

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.protein_name, str(pdb_path))
        self._apply_bfactors(structure, residue_values, default=default_bfactor)
        self._save_pdb(structure, out_path)

        print(f"✓  RMSF-coloured PDB saved → {out_path}  "
              f"({len(residue_values)} residues mapped)")
        return out_path

    # ------------------------------------------------------------------
    # 12. Colour PDB by network centrality metric
    # ------------------------------------------------------------------

    # All centrality metrics supported and the NetworkX function for each
    _CENTRALITY_FUNCTIONS: dict = {
        "degree":       nx.degree_centrality,
        "betweenness":  nx.betweenness_centrality,
        "closeness":    nx.closeness_centrality,
        "eigenvector":  lambda G: nx.eigenvector_centrality(G, max_iter=1000),
    }

    def colour_pdb_by_centrality(
        self,
        pdb_file: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        metric: str = "betweenness",
        output_file: Optional[str] = None,
        default_bfactor: float = 0.0,
        residue_offset: int = 1,
    ) -> Path:
        """
        Replace the B-factor column of a PDB with a per-residue network
        centrality score and write a new PDB.

        The graph produced by :meth:`correlation_network` uses 0-based node
        indices that correspond to residue positions in the analysis group
        (typically Backbone).  ``residue_offset`` maps node index → PDB
        residue number (default ``+1`` for standard 1-indexed PDB files).

        The output is ready for structural visualisation.  In PyMOL::

            spectrum b, yellow_green_blue   # or any diverging palette

        Supported metrics
        -----------------
        ``"degree"``
            Fraction of all possible edges this node has.
        ``"betweenness"``
            How often this node lies on the shortest path between other
            pairs — a good proxy for allosteric communication hubs.
        ``"closeness"``
            Inverse of the mean shortest path to all other nodes.
        ``"eigenvector"``
            Influence score accounting for the centrality of neighbours.

        Parameters
        ----------
        pdb_file : str, optional
            Path to the input PDB.  Defaults to
            ``average_structure_{protein_name}.pdb``.
        graph : networkx.Graph, optional
            Correlation network returned by :meth:`correlation_network`.
            If ``None``, raises ``ValueError`` — the graph must be
            computed first.
        metric : str, optional
            Centrality metric to use.  One of ``"degree"``,
            ``"betweenness"``, ``"closeness"``, ``"eigenvector"``.
            Defaults to ``"betweenness"``.
        output_file : str, optional
            Path for the output PDB.  Defaults to
            ``bfactor_{metric}_{protein_name}.pdb`` in ``work_dir``.
        default_bfactor : float, optional
            B-factor assigned to residues with no graph node (e.g. residues
            that fell below the correlation threshold).  Defaults to ``0.0``.
        residue_offset : int, optional
            Value added to each 0-based node index to obtain the PDB
            residue sequence number.  Defaults to ``1``.  If your PDB
            starts at a different residue number (e.g. 10), set this to
            ``10``.

        Returns
        -------
        Path
            Path to the written PDB file.

        Raises
        ------
        ValueError
            If ``graph`` is ``None`` or ``metric`` is not recognised.
        """
        if graph is None:
            raise ValueError(
                "A NetworkX graph must be supplied via the `graph` parameter.  "
                "Run correlation_network() first."
            )
        if metric not in self._CENTRALITY_FUNCTIONS:
            raise ValueError(
                f"Unknown metric '{metric}'.  "
                f"Choose from: {list(self._CENTRALITY_FUNCTIONS)}"
            )

        pdb_path = Path(pdb_file) if pdb_file else (
            self.work_dir / f"average_structure_{self.protein_name}.pdb"
        )
        out_path = Path(output_file) if output_file else (
            self.work_dir / f"bfactor_{metric}_{self.protein_name}.pdb"
        )

        # Compute centrality: {node_index: score}
        centrality_fn = self._CENTRALITY_FUNCTIONS[metric]
        node_scores: dict[int, float] = centrality_fn(graph)

        # Map 0-based node indices to PDB residue numbers
        residue_values = {
            node + residue_offset: float(score)
            for node, score in node_scores.items()
        }

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.protein_name, str(pdb_path))
        self._apply_bfactors(structure, residue_values, default=default_bfactor)
        self._save_pdb(structure, out_path)

        min_v = min(node_scores.values())
        max_v = max(node_scores.values())
        print(
            f"✓  {metric.capitalize()} centrality PDB saved → {out_path}  "
            f"({len(node_scores)} nodes  |  range {min_v:.4f} – {max_v:.4f})"
        )
        return out_path


# ---------------------------------------------------------------------------
# Quick sanity check / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demonstrates the full pipeline.
    # Replace values below with paths relevant to your own system.

    custom_groups = SelectionGroups(
        center="1",          # Protein – centre for PBC removal
        output="0",          # System  – what gets written to the trajectory
        fit="4",             # Backbone – used for rot+trans fitting
        rmsf="4",            # Backbone – RMSF per residue
        rg="1",              # Protein  – radius of gyration
        covar_fit="4",       # Backbone – fit reference for covar
        covar_analysis="4",  # Backbone – atoms decomposed in PCA
    )

    sim = GromacsAnalysis(
        md_name="md_production",
        protein_name="MyProtein",
        work_dir=".",
        groups=custom_groups,
        # index_file="index.ndx",  # uncomment if you have a custom index
    )

    # --- GROMACS pipeline ---
    sim.nopbc_and_fit()
    sim.essential_dynamics(time_unit="ns")
    sim.covariance_analysis(time_unit="ns", fit=True, use_pbc=False, last=10)

    # --- DCCM pipeline ---
    corr = sim.covariance_to_correlation()
    sim.plot_dccm(corr=corr, cmap="bwr")
    G = sim.correlation_network(corr=corr, threshold=0.3)
    sim.plot_correlation_network(G)

    # --- PCA projection + free energy landscape ---
    # Project onto PC1 and PC2 (default); increase last= for more PCs
    pc_data = sim.project_pca(first=1, last=2, time_unit="ns")

    # Compute landscape at 300 K with 2 Å bin width and light smoothing
    landscape = sim.free_energy_landscape(
        pc_data=pc_data,
        bin_width=2.0,
        sigma=1.5,
        temperature=300.0,
    )

    # Produce 3-D FEL + gradient magnitude figures
    sim.plot_free_energy_3d(landscape, cmap_G="viridis", cmap_grad="plasma")

    # --- Compare two systems: compute landscapes separately then plot ---
    # sim2  = GromacsAnalysis(md_name="md_production", protein_name="SystemB",
    #                          work_dir="/data/simulations/run2")
    # pc2   = sim2.project_pca()
    # land2 = sim2.free_energy_landscape(pc_data=pc2)
    # sim2.plot_free_energy_3d(land2)

    # --- PDB colouring ---
    # Colour by RMSF from the XVG produced by essential_dynamics()
    sim.colour_pdb_by_rmsf(pdb_file="structure.pdb")

    # Colour by betweenness centrality using the network graph computed above
    sim.colour_pdb_by_centrality(pdb_file="structure.pdb", graph=G,
                                  metric="betweenness")