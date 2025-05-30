#!/bin/bash
#SBATCH --output=logs/finetune-report.txt
#SBATCH --error=logs/finetune-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100l:2

module load cuda/12.1.1
module load OpenSSL/1.1
module load libffi/3.2.1

source ~/Projects/my_env/bin/activate

python /home/mila/s/soroush.omranpour/scratch/OmniVLMPlus/finetune.py --token_reduction="$1" --reduction_factor="$2" --image_pe="$3" --num_workers="$4" --image_size=224 --batch_size=128
