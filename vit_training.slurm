#!/usr/bin/env bash
#SBATCH --job-name=vit_training             # Job name
#SBATCH --mem=32G                           # Memory per node
#SBATCH --time=8:00:00                      # Time limit
#SBATCH --account=gts-smussmann3            # Account name
#SBATCH --qos=embers                        # QoS
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --output=logs/vit_training_%j.log   # Output log file with job ID
#SBATCH --error=logs/vit_training_%j.err    # Error log file with job ID

# Load Python 3.10 module
module load python/3.10.10

# Navigate to the project directory
cd ~/scratch/meadow

# Activate the meadowenv virtual environment
source ~/scratch/meadowenv/bin/activate

# Run the training script
python train.py \
    --model-name vit \
    --model-variant B_16 \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 64 \
    --early-stopping-patience 7
