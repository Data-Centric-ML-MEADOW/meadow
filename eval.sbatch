#!/bin/bash
#SBATCH -J eval
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

# srun python3 eval.py --checkpoint-path checkpoints/seeded_experts/resnet-50_20241202-100959_lr2.00e-01_bs128[bagging1].ckpt
srun python3 eval_bagging.py --checkpoints-folder-path checkpoints/seeded_experts/
# srun python3 eval_moe.py --checkpoint-path checkpoints/MoE5_resnet-50_resnet-domain-50_20241203-154219_lr1.00e-03_bs128[bagging_steep_delayExpert_e30_es10_learnMapper].ckpt