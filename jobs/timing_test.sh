#!/bin/bash
#SBATCH --job-name=timing_test
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out

# Short timing test: trains the attention-free GRU on `passive` for 20 minutes,
# then lists the checkpoint files with their save times (one per 1000 steps).
# Submit to a GPU partition with: sbatch --partition=gpu --gpus=1 jobs/timing_test.sh

module reset
module load miniconda
conda activate mtl

cd ~/mtl-transformations

echo "Node: $(hostname)"
python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"

timeout 20m python -u seq2seq.py passive timing_${SLURM_JOB_PARTITION} GRU 0 0.001 256

ls -l --time-style=full-iso models/timing_${SLURM_JOB_PARTITION}_GRU_0_0.001_256_*/
