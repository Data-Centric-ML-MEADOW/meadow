#!/bin/bash
#SBATCH -JResNet152
#SBATCH -N1 --gres=gpu:H100:2 --ntasks-per-node=2
#SBATCH --mem-per-gpu=20G
#SBATCH -t04:00:00
#SBATCH -osbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python train.py \
    --model-name resnet-tfms \
    --model-variant 152 \
    --lr 1e-4 \
    --batch-size 64 \
    --early-stopping-patience 7
