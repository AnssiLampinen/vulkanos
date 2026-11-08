#!/bin/bash
# sbatch scripts/slurm.sh [gpu_run.py args], e.g. sbatch scripts/slurm.sh --train
# Change partition / time / gres to what the school cluster uses (sinfo lists partitions).
#SBATCH --job-name=vulkanos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=vulkanos_%j.log
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate
python scripts/gpu_run.py "$@"
