#!/usr/bin/env bash
##SBATCH -J vit32_unfrozen
#SBATCH -N1 --gres=gpu:L40S:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o sbatch_logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kkang68@gatech.edu

# Load Python 3.10 module
module load python/3.10.10

# Navigate to the project directory
cd ~/scratch/meadow

# Activate the meadowenv virtual environment
source ~/scratch/meadowenv/bin/activate

# Run the training script
python train.py \
    --model-name vit \
    --model-variant 16 \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 64 \
    --early-stopping-patience 7 \
    --no-freeze-backbone
