#!/bin/bash
#SBATCH -J ResNet50-tfms
#SBATCH -N1 --gres=gpu:H100:4 --ntasks-per-node=4 --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH -t 04:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python train.py \
    --model-name resnet-tfms \
    --model-variant 50 \
    --lr 1e-4 \
    --batch-size 128 \
    --early-stopping-patience 7
