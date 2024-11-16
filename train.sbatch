#!/bin/bash
#SBATCH -J r50s3e40
#SBATCH -N1 --gres=gpu:L40S:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env

srun python train.py \
    --model-name resnet \
    --model-variant 50 \
    --epochs 40 \
    --lr 0.2 \
    --batch-size 128 \
    --early-stopping-patience 7 \
    --ensemble-type snapshot-pl \
    --num-estimators 3 \
    --misc-desc 'e40'

# srun python train.py \
#     --model-name resnet-domain \
#     --model-variant 50 \
#     --epochs 30 \
#     --lr 5e-3 \
#     --batch-size 128 \
#     --early-stopping-patience 5 \
#     --misc-desc 'e30_estop5_real'
