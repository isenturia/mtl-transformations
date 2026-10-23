#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out

# Trains one attention-free GRU (GRU, attention 0, lr 0.001, hidden 256) until
# early stopping. Pass the dataset name, e.g.:
#   sbatch jobs/train.sh passive
#   sbatch jobs/train.sh passive_question_first
#   sbatch jobs/train.sh passive_question_main
# Running the same command again trains another model (_1, _2, ...) with a new seed.

DATASET=$1

module reset
module load miniconda
conda activate mtl

cd ~/mtl-transformations

echo "Dataset: $DATASET"
echo "Node: $(hostname)"
python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"

python -u seq2seq.py $DATASET $DATASET GRU 0 0.001 256

ls -ltr models/${DATASET}_GRU_0_0.001_256_*/
