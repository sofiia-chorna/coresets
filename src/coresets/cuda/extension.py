from __future__ import annotations

import os
import subprocess

import torch
from torch.utils.cpp_extension import load

# module level singletons, the first ensure_compiled call builds the extension and later calls no op
_compiled = False
_extension = None


# idempotent
def ensure_compiled() -> None:
    global _compiled, _extension

    if _compiled:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run on a GPU node (#SBATCH --gres=gpu:1).")

    # auto detect CUDA_HOME from nvcc if the env var is not set
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

    # the kernels live next to this file under kernels
    here = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.join(here, "kernels")
    sources = [
        os.path.join(kernels_dir, name)
        for name in (
            "warp.cu",
            "argmax.cu",
            "commit.cu",
            "simulate.cu",
            "multi.cu",
            "bindings.cu",
        )
    ]

    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError(f"CUDA sources missing under {kernels_dir}/: {missing}")

    # first compile takes about 15 seconds, later runs reuse the cache under TORCH_EXTENSIONS_DIR
    print("Compiling custom CUDA FPS kernels... (first time, ~15 s)")

    # the include path is needed for the shared header
    try:
        _extension = load(
            name="coreset_kernels",
            sources=sources,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            extra_include_paths=[kernels_dir],
            verbose=False,
        )
        _compiled = True
        print("Compilation finished. Kernel loaded.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile CUDA kernels: {e}. "
            f"CUDA_HOME={os.environ.get('CUDA_HOME', 'NOT SET')}"
        ) from e


# compiles on first call
def get_ext():
    ensure_compiled()
    return _extension


# compile ahead of timing sensitive code
def warmup_fast_fps() -> None:
    ensure_compiled()
