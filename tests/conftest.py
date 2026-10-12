# only test_bisect_search.py is a pytest module, the rest are gpu scripts with their own cli
collect_ignore = [
    "test_determinism.py",
    "test_fps_sharded.py",
    "test_invariants.py",
    "parity_gpu.py",
    "parity_gpu_sharded.py",
]
