#!/bin/bash
#SBATCH --job-name=accrete-f64cert
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_f64cert_%j.out
#
# Float64 separation/coverage certificate of the madnext GROUND-TRUTH hierarchy
# (default path, max_k=20). Answers: does the f32-constructed hierarchy satisfy
# the paper's strict sep > cov at every level when measured exactly?
set -uo pipefail
MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
MANIFEST=$MAD_CORE/accrete/runs/madnext/manifest.txt
IDS=${1:-$CORESETS/logs/madnext_gt/out/coreset_global_ids.npy}

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

echo "############ f64 certificate  ids=$IDS  $(date -Is) ############"
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/f64_certificate.py" --manifest "$MANIFEST" --ids "$IDS"
echo "############ rc=$? done $(date -Is) ############"
