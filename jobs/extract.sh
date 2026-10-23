#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out

# Runs extract.py on one trained model: saves the .test CSV and prints
# .test and .gen accuracy. Pass the dataset and the run number, e.g.:
#   sbatch jobs/extract.sh passive 0

DATASET=$1
RUN=$2

module reset
module load miniconda
conda activate mtl

cd ~/mtl-transformations

python -u extract.py models/${DATASET}_GRU_0_0.001_256_${RUN} $DATASET
