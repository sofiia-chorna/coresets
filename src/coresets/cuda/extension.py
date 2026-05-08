"""JIT-compiled CUDA extension shared across the ``cuda`` subpackage.

Built from sources in ``kernels/`` on first use; cached afterwards. All
:mod:`coresets.cuda` modules go through :func:`get_ext`.

Exposed kernels:
  fps_update_cuda                : dense Hausdorff update (1 warp per row)
  voronoi_update_cuda            : sparse Voronoi-skip update
  argmax_phase1 / phase2         : two-phase device argmax with deterministic tie-break
  simulate_fps_voronoi           : full FPS sim loop in C++ with Voronoi skip
  commit_center_cuda             : on-device commit + bookkeeping
  compute_dSL_quarter_cuda       : per-step distance bound update
  reduce_and_commit_multi_*      : multi-GPU commits
  simulate_fps_voronoi_multi_*   : multi-GPU sim loops
"""

from __future__ import annotations

import os
import subprocess

import torch
from torch.utils.cpp_extension import load

# module-level singletons - first call to ensure_compiled() builds the extension; later calls no-op.
_compiled = False
_extension = None


def ensure_compiled() -> None:
    """Compile/load the CUDA extension on first use. Idempotent."""
    global _compiled, _extension
    if _compiled:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run on a GPU node (#SBATCH --gres=gpu:1).")

    # auto-detect CUDA_HOME from `which nvcc` if the env var isn't set.
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

    # the kernels live next to this file under kernels/.
    here = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.join(here, "kernels")
    sources = [
        os.path.join(kernels_dir, name)
        for name in ("warp.cu", "argmax.cu", "commit.cu", "simulate.cu", "multi.cu", "bindings.cu")
    ]
    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError(f"CUDA sources missing under {kernels_dir}/: {missing}")

    # first-time compile is ~15 s; subsequent runs reuse the cache under TORCH_EXTENSIONS_DIR.
    print("Compiling custom CUDA FPS kernels... (first time, ~15 s)")
    try:
        _extension = load(
            name="coreset_kernels",
            sources=sources,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            extra_include_paths=[kernels_dir],  # for #include "_shared.cuh"
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
    """Return the compiled extension (compiles on first call)."""
    ensure_compiled()
    return _extension


def warmup_fast_fps() -> None:
    """Compile the extension ahead of timing-sensitive code."""
    ensure_compiled()
