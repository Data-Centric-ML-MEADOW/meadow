#!/bin/bash
#SBATCH -J m34e30SDEL
#SBATCH -N1 --gres=gpu:L40S:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nthomas90@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc
module load anaconda3

conda activate dml_env

srun python train_moe.py \
  --expert-checkpoints-folder checkpoints/seeded_experts \
  --domain-mapper-checkpoint-path checkpoints/resnet-domain-50_20241202-105204_lr5.00e-03_bs128[e30_estop5_real].ckpt \
  --epochs 30 \
  --lr 1e-3 \
  --batch-size 128 \
  --early-stopping-patience 10 \
  --learn-domain-mapper \
  --misc-desc "bagging_steep_delayExpert_e30_es10_learnMapper"
