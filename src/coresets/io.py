from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from ._version import __version__

SCHEMA_VERSION = 1

# provenance runs parallel to coreset.npy and resolves each pick to its source id and row
PROVENANCE_DTYPE = np.dtype([("source_id", np.uint16), ("source_row", np.int64)])

PathLike = Union[str, os.PathLike]
_STEP_DIR_RE = re.compile(r"^(\d{3,})_(.+)$")


# one result in one directory

KNOB_PREFIX = "CORESETS_"


def active_knobs() -> dict:
    # every set variable with the knob prefix, so a result can be reproduced later
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(KNOB_PREFIX)
    }


def save_result(
    out_dir: PathLike,
    selected,
    *,
    epsilons=None,
    n_rows: Optional[int] = None,
    dim: Optional[int] = None,
    backend: str = "",
    walltime_s: Optional[float] = None,
    world_size: int = 1,
    extra: Optional[dict] = None,
    overwrite: bool = False,
) -> Path:
    out = Path(out_dir)
    selected_np = _to_int64_numpy(selected)

    if selected_np.ndim != 1:
        raise ValueError(f"selected must be 1-D, got shape {selected_np.shape}")

    out.mkdir(parents=True, exist_ok=True)

    if (out / "selected.npy").exists() and not overwrite:
        raise FileExistsError(f"{out / 'selected.npy'} exists, pass overwrite=True")

    _save_npy_atomic(out / "selected.npy", selected_np)

    if epsilons is not None:
        epsilons_np = np.asarray(_to_numpy(epsilons), dtype=np.float32)

        if epsilons_np.shape != selected_np.shape:
            raise ValueError(
                f"epsilons shape {epsilons_np.shape} != selected shape {selected_np.shape}"
            )

        _save_npy_atomic(out / "epsilons.npy", epsilons_np)

    meta = {
        "schema": "coresets.result/1",
        "K": int(selected_np.shape[0]),
        "n_rows": int(n_rows) if n_rows is not None else None,
        "dim": int(dim) if dim is not None else None,
        "backend": backend,
        "world_size": int(world_size),
        "walltime_s": float(walltime_s) if walltime_s is not None else None,
        "has_epsilons": epsilons is not None,
        "knobs": active_knobs(),
        "git_commit": _git_commit(Path(__file__).parent),
        "coresets_version": __version__,
        "written_at": _now_iso(),
    }

    if extra:
        meta.update(extra)

    _atomic_write_json(out / "meta.json", meta)

    return out


def load_result(out_dir: PathLike) -> dict:
    out = Path(out_dir)

    selected = np.load(out / "selected.npy")

    epsilons_path = out / "epsilons.npy"
    epsilons = np.load(epsilons_path) if epsilons_path.is_file() else None

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    return {"selected": selected, "epsilons": epsilons, "meta": meta}


def _to_numpy(x):
    # torch tensors on any device, or anything numpy accepts
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _to_int64_numpy(x) -> np.ndarray:
    return np.ascontiguousarray(_to_numpy(x), dtype=np.int64)


# chained runs with a manifest


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# best effort commit hash for provenance, None outside a repo
def _git_commit(cwd: Optional[PathLike] = None) -> Optional[str]:
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


# tmp file in the same dir then rename
def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(content)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False) + "\n")


# create the run directory with a fresh manifest
def init_run(
    results_root: PathLike,
    run_name: str,
    *,
    target_K: int,
    feature_dim: int,
    description: str = "",
    overwrite: bool = False,
) -> Path:
    root = Path(results_root) / run_name

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"run already exists: {root}. Pass overwrite=True to wipe.")

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


# returns the source id, the small uint16 stored in provenance.npy
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


# step files


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


# body of step.json, LATEST is not advanced when success is False
@dataclass
class StepStats:
    max_k: int
    n_anchor: int
    n_candidates: int
    walltime_s: float
    peak_mem_gb: float = 0.0
    max_rel_gap: float = 0.0
    success: bool = True
    parents: Sequence[str] = field(default_factory=tuple)
    candidate_sources: Sequence[str] = field(default_factory=tuple)


# write one step directory and move LATEST on success
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

    # only successful steps move LATEST
    if stats.success:
        _atomic_write_text(run_dir / "LATEST", step_dir.name + "\n")

    return step_dir


# open file handle so np.save does not append .npy to the tmp suffix
def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)

    os.replace(tmp, path)


# accepts the bare step name or the prefixed dir name
def load_step(
    results_root: PathLike,
    run_name: str,
    step_name_or_dir: str,
) -> dict:
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


# load the step named in LATEST
def load_latest(results_root: PathLike, run_name: str) -> dict:
    run_dir = Path(results_root) / run_name
    latest_path = run_dir / "LATEST"

    if not latest_path.is_file():
        raise FileNotFoundError(f"no LATEST at {run_dir}; run has no completed steps")

    name = latest_path.read_text().strip()

    return load_step(results_root, run_name, name)


# step directory names sorted by numeric prefix
def list_steps(results_root: PathLike, run_name: str) -> list[str]:
    run_dir = Path(results_root) / run_name
    out = []

    for p in run_dir.iterdir():
        if p.is_dir() and _STEP_DIR_RE.match(p.name):
            out.append(p.name)

    out.sort()

    return out


# bare name or prefixed dir name, latest match wins on a re-run
def _resolve_step_dir(run_dir: Path, name: str) -> Path:
    if (run_dir / name).is_dir():
        return run_dir / name

    matches = [
        p
        for p in run_dir.iterdir()
        if p.is_dir()
        and _STEP_DIR_RE.match(p.name)
        and _STEP_DIR_RE.match(p.name).group(2) == name
    ]

    if not matches:
        raise FileNotFoundError(f"no step matching {name!r} in {run_dir}")

    matches.sort()

    return matches[-1]


# per level streaming


_PER_LEVEL_HEADER = "k\tsize\tsep\tcov\trel_gap\twalltime_s\n"


# flushes after every line so the file can be tailed during a run
def open_per_level(step_dir: PathLike) -> "_PerLevelWriter":
    return _PerLevelWriter(Path(step_dir) / "per_level.tsv")


class _PerLevelWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

        # line buffered for tailing
        self._fh = open(path, "w", buffering=1)
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


# append one row, writes the header on first call
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
    run_dir = Path(results_root) / run_name
    step_dir = _resolve_step_dir(run_dir, step_name_or_dir)
    path = step_dir / "per_level.tsv"

    write_header = not path.is_file()

    with open(path, "a", buffering=1) as fh:
        if write_header:
            fh.write(_PER_LEVEL_HEADER)

        fh.write(f"{k}\t{size}\t{sep:.10g}\t{cov:.10g}\t{rel_gap:.10g}\t{walltime_s:.6f}\n")
        fh.flush()
