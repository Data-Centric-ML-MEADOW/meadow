#!/bin/bash
#SBATCH -J ss3nfz50
#SBATCH -N1 --gres=gpu:H100:2 --ntasks-per-node=2 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python train.py \
    --model-name resnet \
    --model-variant 34 \
    --epochs 50 \
    --lr 0.2 \
    --batch-size 128 \
    --early-stopping-patience 7 \
    --no-freeze-backbone \
    --ensemble-type snapshot-pl \
    --num-estimators 3 \
    --misc-desc 'e50_nofreeze_2gpu'

