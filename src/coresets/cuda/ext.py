"""JIT-compiled CUDA extension shared across the ``cuda`` subpackage.

The extension is built from ``kernels.cu`` (sibling file) on first use. All
modules in ``coresets.cuda`` access the compiled kernels via :func:`get_ext`,
which compiles on first call and caches the result.

The kernels exposed by the extension:
  - fps_update_cuda        : dense Hausdorff update (1 warp per row)
  - voronoi_update_cuda    : sparse Voronoi-skip update
  - argmax_phase1/_phase2  : two-phase device argmax with deterministic tie-break
  - simulate_fps_voronoi   : full FPS simulation loop in C++ with Voronoi skip
  - commit_center_cuda     : on-device commit + bookkeeping
  - compute_dSL_quarter_cuda
  - reduce_and_commit_multi_kernel{,_sharded}  (multi-GPU)
  - simulate_fps_voronoi_multi{,_sharded}      (multi-GPU)
"""

from __future__ import annotations

import os
import subprocess

import torch
from torch.utils.cpp_extension import load

_compiled = False
_extension = None  # the loaded JIT module; populated on first ensure_compiled()


def ensure_compiled() -> None:
    """Compile/load the CUDA extension on first use. Idempotent."""
    global _compiled, _extension

    if _compiled:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script requires CUDA, but CUDA is not available. "
            "Run on a compute node with a GPU (#SBATCH --gres=gpu:1)."
        )

    if "CUDA_HOME" not in os.environ:
        try:
            nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))
            print(f"Auto-detected CUDA_HOME: {os.environ['CUDA_HOME']}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise OSError(
                "CUDA_HOME unset and nvcc not on PATH. "
                "Load CUDA modules first (e.g. `module load cuda/12.5.1`)."
            ) from e

    here = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.join(here, "kernels")
    sources = [
        os.path.join(kernels_dir, fname)
        for fname in ("warp.cu", "argmax.cu", "commit.cu", "simulate.cu", "multi.cu", "bindings.cu")
    ]
    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError(
            f"CUDA kernel sources missing: {missing}. " f"Expected under {kernels_dir}/."
        )

    print("Compiling custom CUDA FPS kernels… (first time, ~15 s)")
    try:
        _extension = load(
            name="coreset_kernels",
            sources=sources,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            extra_include_paths=[kernels_dir],  # for "#include _shared.cuh"
            verbose=False,
        )
        _compiled = True
        print("Compilation finished. Kernel loaded.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile CUDA kernels: {e}. "
            f"CUDA_HOME={os.environ.get('CUDA_HOME', 'NOT SET')}"
        ) from e


def get_ext():
    """Return the compiled extension module (compiles on first call)."""
    ensure_compiled()
    return _extension


def warmup_fast_fps() -> None:
    """Compile the CUDA extension ahead of timing-sensitive code."""
    ensure_compiled()
