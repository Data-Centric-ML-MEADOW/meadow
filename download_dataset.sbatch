#!/bin/bash
#SBATCH -J download
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o sbatch_logs/download.out

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env

srun python download_dataset.py
