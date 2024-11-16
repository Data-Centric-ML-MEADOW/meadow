#!/bin/bash
#SBATCH -J m34e30SDEL
#SBATCH -N1 --gres=gpu:L40S:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env

srun python train_moe.py \
  --expert-ensemble-checkpoint-path checkpoints/resnet-34_20241113-212612_lr2.00e-01_bs128_snapshot-pl5[e40].ckpt \
  --domain-mapper-checkpoint-path checkpoints/resnet-domain-50_20241114-205057_lr5.00e-03_bs128[e30_estop5_real].ckpt \
  --epochs 30 \
  --lr 1e-3 \
  --batch-size 128 \
  --early-stopping-patience 10 \
  --learn-domain-mapper \
  --misc-desc "steep_delayExpert_e30_es10_learnMapper"
