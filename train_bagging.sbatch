#!/bin/bash
#SBATCH -J bagging5
#SBATCH -N1 --gres=gpu:L40S:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nthomas90@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc
module load anaconda3

conda activate dml_env

srun python train.py \
    --model-name resnet \
    --model-variant 50 \
    --epochs 40 \
    --lr 0.2 \
    --batch-size 128 \
    --early-stopping-patience 7 \
    --misc-desc 'bagging5' \
    --bagging \
    --seed 5

# srun python train.py \
#     --model-name resnet-domain \
#     --model-variant 50 \
#     --epochs 30 \
#     --lr 5e-3 \
#     --batch-size 128 \
#     --early-stopping-patience 5 \
#     --misc-desc 'e30_estop5_real'
