"""Standard results format for chained coreset runs.

Layout::

    results/<run_name>/
    ├── manifest.json   # target_K, data sources, schema
    ├── LATEST          # name of most recent successful step
    └── NNN_<name>/
        ├── coreset.npy        # int64 (K,) — the answer
        ├── provenance.npy     # (source_id uint16, source_row int64)
        ├── per_level.tsv      # streamed: k, size, sep, cov, rel_gap, walltime_s
        └── step.json          # parents, max_k, walltime, peak_mem, max_rel_gap, success

``NNN`` is auto-assigned by :func:`save_step` (000, 001, …); ``name`` is
user-chosen. ``parents`` in ``step.json`` encodes lineage for re-runs/branches.

API: :func:`init_run`, :func:`register_source`, :func:`save_step`,
:func:`load_step`, :func:`load_latest`, :func:`append_per_level`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

SCHEMA_VERSION = 1

# Provenance array runs parallel to coreset.npy: provenance[i] resolves the
# i-th selected point back to (source_id, source_row).
PROVENANCE_DTYPE = np.dtype([("source_id", np.uint16), ("source_row", np.int64)])

PathLike = Union[str, os.PathLike]
_STEP_DIR_RE = re.compile(r"^(\d{3,})_(.+)$")


# --------------------------------------------------------------------------- #
# Manifest / run-level
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(cwd: Optional[PathLike] = None) -> Optional[str]:
    """Best-effort git commit hash for provenance. Returns None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically (tmp file in same dir + os.replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False) + "\n")


def init_run(
    results_root: PathLike,
    run_name: str,
    *,
    target_K: int,
    feature_dim: int,
    description: str = "",
    overwrite: bool = False,
) -> Path:
    """Create ``results_root/<run_name>/`` with a fresh ``manifest.json``.

    Subsequent :func:`save_step` calls append to this run. Use
    :func:`register_source` to record dataset metadata before the first step.
    """
    root = Path(results_root) / run_name
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"run already exists: {root}. Pass overwrite=True to wipe.")
        import shutil  # rmtree is destructive; only reachable via explicit overwrite

        shutil.rmtree(root)
    root.mkdir(parents=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "description": description,
        "target_K": int(target_K),
        "feature_dim": int(feature_dim),
        "data_sources": {},
        "git_commit": _git_commit(),
        "created_at": _now_iso(),
    }
    _atomic_write_json(root / "manifest.json", manifest)
    return root


def register_source(
    results_root: PathLike,
    run_name: str,
    source_name: str,
    *,
    path: Optional[PathLike] = None,
    n_rows: Optional[int] = None,
    sha256: Optional[str] = None,
    extra: Optional[dict] = None,
) -> int:
    """Add a data source to the run's manifest; returns the assigned source_id.

    The id is the small uint16 stored in ``provenance.npy`` — keeps per-row
    bookkeeping cheap while still resolving back to ``(dataset, row)``.
    """
    root = Path(results_root) / run_name
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    sources = manifest["data_sources"]
    if source_name in sources:
        return int(sources[source_name]["id"])

    new_id = max((s["id"] for s in sources.values()), default=-1) + 1
    sources[source_name] = {
        "id": new_id,
        "path": str(path) if path is not None else None,
        "n_rows": int(n_rows) if n_rows is not None else None,
        "sha256": sha256,
        **(extra or {}),
    }
    _atomic_write_json(manifest_path, manifest)
    return new_id


def read_manifest(results_root: PathLike, run_name: str) -> dict:
    return json.loads((Path(results_root) / run_name / "manifest.json").read_text())


# --------------------------------------------------------------------------- #
# Step I/O
# --------------------------------------------------------------------------- #


def _next_step_idx(run_dir: Path) -> int:
    existing = []
    for p in run_dir.iterdir():
        if p.is_dir():
            m = _STEP_DIR_RE.match(p.name)
            if m:
                existing.append(int(m.group(1)))
    return max(existing, default=-1) + 1


def _step_dir(run_dir: Path, step_idx: int, name: str) -> Path:
    return run_dir / f"{step_idx:03d}_{name}"


@dataclass
class StepStats:
    """Summary of a step. Forms the body of ``step.json``.

    Fields:
        max_k: final coreset size produced by this step (``= 2 ** max_k``).
        n_anchor: number of anchor rows fed in.
        n_candidates: number of candidate rows fed in.
        walltime_s: wall-clock time of the step (seconds).
        peak_mem_gb: peak GPU memory used (informational).
        max_rel_gap: worst per-level ``sep_sq / cov_sq - 1`` seen in this step.
        success: ``False`` if the step aborted; ``LATEST`` is **not** advanced
            in that case.
        parents: step names this step builds on (``[]`` for a from-scratch step).
        candidate_sources: data-source names whose rows formed the candidate pool.
    """

    max_k: int
    n_anchor: int
    n_candidates: int
    walltime_s: float
    peak_mem_gb: float = 0.0
    max_rel_gap: float = 0.0
    success: bool = True
    parents: Sequence[str] = field(default_factory=tuple)
    candidate_sources: Sequence[str] = field(default_factory=tuple)


def save_step(
    results_root: PathLike,
    run_name: str,
    step_name: str,
    *,
    coreset: np.ndarray,
    provenance: Optional[np.ndarray] = None,
    stats: StepStats,
    started_at: Optional[str] = None,
    step_idx: Optional[int] = None,
) -> Path:
    """Atomically write one step's outputs and update LATEST.

    Step dir is ``results/<run_name>/NNN_<step_name>/`` — ``NNN`` is
    auto-assigned (or supplied via ``step_idx`` to overwrite).

    ``provenance`` (parallel to ``coreset``) resolves each pick back to its
    *original* dataset+row. Strongly recommended — without it, "leftover
    from prev step" isn't computable later.
    """
    if coreset.ndim != 1 or coreset.dtype != np.int64:
        raise ValueError(
            f"coreset must be 1-D int64; got shape={coreset.shape}, dtype={coreset.dtype}"
        )
    if provenance is not None:
        if provenance.dtype != PROVENANCE_DTYPE:
            raise ValueError(
                f"provenance must have dtype {PROVENANCE_DTYPE}; got {provenance.dtype}"
            )
        if provenance.shape != coreset.shape:
            raise ValueError(
                f"provenance shape {provenance.shape} != coreset shape {coreset.shape}"
            )

    run_dir = Path(results_root) / run_name
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"no manifest at {run_dir}; call init_run first.")

    if step_idx is None:
        step_idx = _next_step_idx(run_dir)
    step_dir = _step_dir(run_dir, step_idx, step_name)
    step_dir.mkdir(parents=True, exist_ok=True)

    _save_npy_atomic(step_dir / "coreset.npy", coreset)
    if provenance is not None:
        _save_npy_atomic(step_dir / "provenance.npy", provenance)

    finished = _now_iso()
    step_meta = {
        "name": step_name,
        "step_idx": int(step_idx),
        "parents": list(stats.parents),
        "candidate_sources": list(stats.candidate_sources),
        "max_k": int(stats.max_k),
        "n_anchor": int(stats.n_anchor),
        "n_candidates": int(stats.n_candidates),
        "coreset_size": int(coreset.shape[0]),
        "walltime_s": float(stats.walltime_s),
        "peak_mem_gb": float(stats.peak_mem_gb),
        "max_rel_gap": float(stats.max_rel_gap),
        "success": bool(stats.success),
        "started_at": started_at,
        "finished_at": finished,
    }
    _atomic_write_json(step_dir / "step.json", step_meta)

    # Only successful steps move LATEST — keeps the last-good pointer intact
    # when a half-finished step is saved with success=False.
    if stats.success:
        _atomic_write_text(run_dir / "LATEST", step_dir.name + "\n")
    return step_dir


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    # Write via an open file handle so np.save doesn't auto-append ``.npy``
    # to our tmp suffix and break the atomic rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def load_step(
    results_root: PathLike,
    run_name: str,
    step_name_or_dir: str,
) -> dict:
    """Read a step. Accepts either the bare name (``after_b1``) or the
    prefixed dir name (``002_after_b1``).
    """
    run_dir = Path(results_root) / run_name
    step_dir = _resolve_step_dir(run_dir, step_name_or_dir)
    coreset = np.load(step_dir / "coreset.npy")
    prov_path = step_dir / "provenance.npy"
    provenance = np.load(prov_path) if prov_path.is_file() else None
    step_meta = json.loads((step_dir / "step.json").read_text())
    return {
        "coreset": coreset,
        "provenance": provenance,
        "meta": step_meta,
        "dir": step_dir,
    }


def load_latest(results_root: PathLike, run_name: str) -> dict:
    """Load the step named in the run's ``LATEST`` file."""
    run_dir = Path(results_root) / run_name
    latest_path = run_dir / "LATEST"
    if not latest_path.is_file():
        raise FileNotFoundError(f"no LATEST at {run_dir}; run has no completed steps")
    name = latest_path.read_text().strip()
    return load_step(results_root, run_name, name)


def list_steps(results_root: PathLike, run_name: str) -> list[str]:
    """Return completed step directory names sorted by numeric prefix."""
    run_dir = Path(results_root) / run_name
    out = []
    for p in run_dir.iterdir():
        if p.is_dir() and _STEP_DIR_RE.match(p.name):
            out.append(p.name)
    out.sort()
    return out


def _resolve_step_dir(run_dir: Path, name: str) -> Path:
    """Accept either ``002_after_b1`` or just ``after_b1`` (latest match wins)."""
    if (run_dir / name).is_dir():
        return run_dir / name
    matches = [
        p
        for p in run_dir.iterdir()
        if p.is_dir() and _STEP_DIR_RE.match(p.name) and _STEP_DIR_RE.match(p.name).group(2) == name
    ]
    if not matches:
        raise FileNotFoundError(f"no step matching {name!r} in {run_dir}")
    matches.sort()
    return matches[-1]  # latest wins on re-run with the same step name


# --------------------------------------------------------------------------- #
# Per-level streaming
# --------------------------------------------------------------------------- #


_PER_LEVEL_HEADER = "k\tsize\tsep\tcov\trel_gap\twalltime_s\n"


def open_per_level(step_dir: PathLike) -> "_PerLevelWriter":
    """Streaming per-level TSV writer for an in-progress step.

    Flushes after every line — safely tail-able from another process during
    multi-day runs.

    Usage::

        step_dir.mkdir(parents=True, exist_ok=True)
        with open_per_level(step_dir) as plog:
            for k, size, sep, cov, rel_gap, dt in level_iter():
                plog.write(k, size, sep, cov, rel_gap, dt)
    """
    return _PerLevelWriter(Path(step_dir) / "per_level.tsv")


class _PerLevelWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = open(path, "w", buffering=1)  # line-buffered for tailing
        self._fh.write(_PER_LEVEL_HEADER)

    def write(
        self,
        k: int,
        size: int,
        sep: float,
        cov: float,
        rel_gap: float,
        walltime_s: float,
    ) -> None:
        self._fh.write(f"{k}\t{size}\t{sep:.10g}\t{cov:.10g}\t{rel_gap:.10g}\t{walltime_s:.6f}\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "_PerLevelWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def append_per_level(
    results_root: PathLike,
    run_name: str,
    step_name_or_dir: str,
    *,
    k: int,
    size: int,
    sep: float,
    cov: float,
    rel_gap: float,
    walltime_s: float,
) -> None:
    """Append one row to a step's per_level.tsv (creates with header on first
    call). Prefer :func:`open_per_level` for hot-loop writes."""
    run_dir = Path(results_root) / run_name
    step_dir = _resolve_step_dir(run_dir, step_name_or_dir)
    path = step_dir / "per_level.tsv"
    write_header = not path.is_file()
    with open(path, "a", buffering=1) as fh:
        if write_header:
            fh.write(_PER_LEVEL_HEADER)
        fh.write(f"{k}\t{size}\t{sep:.10g}\t{cov:.10g}\t{rel_gap:.10g}\t{walltime_s:.6f}\n")
        fh.flush()
