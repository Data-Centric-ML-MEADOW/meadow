#!/bin/bash
#SBATCH -JPretrainedResNet
#SBATCH -N1 --gres=gpu:V100:2 --ntasks-per-node=2
#SBATCH --mem-per-gpu=24G
#SBATCH -t05:00:00
#SBATCH -osbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rso31@gatech.edu

cd $HOME/scratch/meadow
source ~/.bashrc

conda activate dml_env
srun python train.py \
    --model-name resnet \
    --model-variant 152 \
    --lr 1e-4 \
    --batch-size 128
