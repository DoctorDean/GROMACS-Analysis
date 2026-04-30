"""
sim_prep/checkpoint.py
======================
Persistent checkpoint system for :class:`~sim_prep.base.SimulationPrepper`.

Why this exists
---------------
MD simulation preparation pipelines on HPC clusters routinely fail
mid-run due to wall-time limits, node failures, or storage quota issues.
Without checkpointing the user must either re-run the entire preparation
from the start or manually reconstruct state — both are error-prone.

This module provides a :class:`CheckpointMixin` that any
``SimulationPrepper`` subclass can inherit to gain:

- **Automatic state persistence** — every pipeline step writes a JSON
  checkpoint file on completion.
- **Resume detection** — ``resume_from_checkpoint()`` reads the file
  and skips all steps that already completed.
- **Step guard** — the ``@checkpoint_step`` decorator wraps any method
  so it is skipped (with a log message) if already marked complete.

Checkpoint file format
----------------------
A plain JSON file (``.sim_checkpoint.json``) written to ``working_dir``:

.. code-block:: json

    {
        "protein_name": "hsp90",
        "sim_type": "apo",
        "completed_steps": ["clean_pdb_file", "protein_pdb2gmx", "set_new_box"],
        "last_updated": "2025-03-12T14:32:01"
    }

Usage
-----
Inherit ``CheckpointMixin`` alongside your prepper class:

    from sim_prep.checkpoint import CheckpointMixin, checkpoint_step

    class ApoSimPrepper(CheckpointMixin, SimulationPrepper):
        ...

        @checkpoint_step
        def protein_pdb2gmx(self):
            ...

Or use the mixin's ``run_step()`` method to manually guard a call:

    sim.run_step("solvate", sim.solvate)

The simplest usage — call ``resume_from_checkpoint()`` at the top of
your pipeline script and then call every step normally.  Steps that
already completed are silently skipped:

    sim = ApoSimPrepper(...)
    sim.validate_config()
    sim.assign_attributes()
    sim.resume_from_checkpoint()    # loads state; marks completed steps

    sim.clean_pdb_file()            # skipped if already done
    sim.protein_pdb2gmx()           # skipped if already done
    sim.set_new_box()               # runs from here if this was the failure point
    ...
"""

from __future__ import annotations

import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

#: Name of the checkpoint file written to the working directory
CHECKPOINT_FILENAME = ".sim_checkpoint.json"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def checkpoint_step(method: Callable) -> Callable:
    """
    Decorator that skips a pipeline method if it is already recorded as
    complete in the instance's checkpoint.

    Apply to any ``SimulationPrepper`` method that should be skippable
    on resume.  The method name is used as the checkpoint key.

    Example
    -------
    ::

        @checkpoint_step
        def protein_pdb2gmx(self):
            ...   # only runs if "protein_pdb2gmx" not in completed_steps
    """
    @functools.wraps(method)
    def wrapper(self: "CheckpointMixin", *args: Any, **kwargs: Any) -> Any:
        step_name = method.__name__
        if hasattr(self, "_completed_steps") and step_name in self._completed_steps:
            print(f"⏭  Skipping '{step_name}' (already completed)")
            return None
        result = method(self, *args, **kwargs)
        if hasattr(self, "_mark_complete"):
            self._mark_complete(step_name)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------

class CheckpointMixin:
    """
    Mixin that adds checkpoint persistence and resume to any
    ``SimulationPrepper`` subclass.

    Attributes
    ----------
    _completed_steps : set[str]
        Set of method names that have completed successfully.
    _checkpoint_path : Path
        Path to the JSON checkpoint file.

    Notes
    -----
    ``CheckpointMixin`` must appear **before** ``SimulationPrepper`` in
    the MRO so its ``__init__`` can initialise state before the base
    class runs:

        class ApoSimPrepper(CheckpointMixin, SimulationPrepper):
            ...
    """

    def __init__(self, **kwargs: Any) -> None:
        self._completed_steps: set[str] = set()
        # working_dir may not be set yet if called before SimulationPrepper.__init__
        # We resolve the path lazily in _ensure_checkpoint_path()
        self._checkpoint_path: Optional[Path] = None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_checkpoint_path(self) -> Path:
        """Resolve the checkpoint path once working_dir is available."""
        if self._checkpoint_path is None:
            # working_dir is set by SimulationPrepper.__init__
            self._checkpoint_path = (
                getattr(self, "working_dir", Path.cwd()) / CHECKPOINT_FILENAME
            )
        return self._checkpoint_path

    def _load_checkpoint(self) -> dict[str, Any]:
        """Read and return the checkpoint JSON, or an empty dict if absent."""
        path = self._ensure_checkpoint_path()
        if path.exists():
            with path.open() as fh:
                return json.load(fh)
        return {}

    def _save_checkpoint(self) -> None:
        """Write current state to the checkpoint JSON file."""
        path = self._ensure_checkpoint_path()
        data = {
            "protein_name":    getattr(self, "protein_name", "unknown"),
            "sim_type":        type(self).__name__,
            "completed_steps": sorted(self._completed_steps),
            "last_updated":    datetime.now().isoformat(timespec="seconds"),
        }
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)

    def _mark_complete(self, step_name: str) -> None:
        """Record a step as complete and persist to disk."""
        self._completed_steps.add(step_name)
        self._save_checkpoint()
        print(f"✓  Checkpoint saved: '{step_name}' complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self) -> list[str]:
        """
        Load the checkpoint file from the working directory (if it exists)
        and populate ``_completed_steps`` so that subsequent pipeline calls
        skip already-finished steps.

        Returns
        -------
        list[str]
            Sorted list of step names that were marked as complete.
            Empty list if no checkpoint file was found.

        Example
        -------
        ::

            sim = ApoSimPrepper(protein_name="hsp90", ...)
            sim.validate_config()
            sim.assign_attributes()
            sim.resume_from_checkpoint()  # load prior state
            sim.clean_pdb_file()          # skipped if done
            sim.protein_pdb2gmx()         # skipped if done
            sim.set_new_box()             # resumes here if this failed before
        """
        data = self._load_checkpoint()
        if not data:
            print("  No checkpoint file found — starting from scratch")
            return []

        completed = data.get("completed_steps", [])
        self._completed_steps = set(completed)

        print(
            f"✓  Checkpoint loaded from {self._ensure_checkpoint_path().name}\n"
            f"   Simulation  : {data.get('sim_type', 'unknown')} / "
            f"{data.get('protein_name', 'unknown')}\n"
            f"   Last updated: {data.get('last_updated', 'unknown')}\n"
            f"   Completed ({len(completed)}): {', '.join(completed)}"
        )
        return completed

    def run_step(self, step_name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run a pipeline step with checkpoint guarding, without requiring
        the ``@checkpoint_step`` decorator on the method.

        Use this when you cannot modify the method definition (e.g. for
        methods inherited from a third-party base class).

        Parameters
        ----------
        step_name : str
            Name to use as the checkpoint key.
        fn : callable
            The method or function to call.
        *args, **kwargs
            Passed through to ``fn``.

        Returns
        -------
        Any
            Return value of ``fn``, or ``None`` if the step was skipped.

        Example
        -------
        ::

            sim.run_step("solvate", sim.solvate)
            sim.run_step("minimise", sim.minimise_system, maxwarn=1)
        """
        if step_name in self._completed_steps:
            print(f"⏭  Skipping '{step_name}' (already completed)")
            return None
        result = fn(*args, **kwargs)
        self._mark_complete(step_name)
        return result

    def reset_checkpoint(self, steps: Optional[list[str]] = None) -> None:
        """
        Remove completed-step records from the checkpoint.

        Parameters
        ----------
        steps : list[str], optional
            Specific step names to un-mark.  If ``None``, all records
            are cleared (full reset).

        Example
        -------
        Force a single step to re-run::

            sim.reset_checkpoint(["nvt_equilibration"])

        Start the whole pipeline from scratch::

            sim.reset_checkpoint()
        """
        if steps is None:
            self._completed_steps.clear()
            path = self._ensure_checkpoint_path()
            if path.exists():
                path.unlink()
            print("✓  Checkpoint fully reset")
        else:
            for s in steps:
                self._completed_steps.discard(s)
            self._save_checkpoint()
            print(f"✓  Reset checkpoint for: {', '.join(steps)}")

    def checkpoint_status(self) -> None:
        """Print a summary of completed and pending steps."""
        data = self._load_checkpoint()
        if not data:
            print("  No checkpoint file found.")
            return

        completed = set(data.get("completed_steps", []))
        print(f"\n{'─'*50}")
        print(f"  Checkpoint status – {data.get('protein_name', '?')}")
        print(f"  Last updated: {data.get('last_updated', '?')}")
        print(f"{'─'*50}")
        for step in completed:
            print(f"  ✓  {step}")
        print(f"{'─'*50}\n")
