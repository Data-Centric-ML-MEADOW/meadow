#!/bin/bash
#SBATCH -J rnet18-bagged
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH -t 06:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python train.py \
    --model-name resnet \
    --model-variant 18 \
    --lr 1e-3 \
    --batch-size 128 \
    --early-stopping-patience 7 \
    --ensemble-type bagging \
    --num-estimators 7
