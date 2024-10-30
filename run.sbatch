#!/bin/bash
#SBATCH -JPretrainedResNet
#SBATCH -N1 --gres=gpu:H100:4 --ntasks-per-node=4
#SBATCH --mem-per-gpu=64G
#SBATCH -t02:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python main.py --resnet-variant 152 --no-freeze-backbone
# srun python main.py --resnet-variant 152 --lr 1e-4
