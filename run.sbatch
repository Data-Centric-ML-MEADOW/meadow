#!/bin/bash
#SBATCH -JPretrainedResNet
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH -t10:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python main.py
