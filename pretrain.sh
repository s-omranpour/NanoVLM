#!/bin/bash
#SBATCH --output=logs/pretrain-report.txt
#SBATCH --error=logs/pretrain-error.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100l:1

module load cuda/12.1.1
module load OpenSSL/1.1
module load libffi/3.2.1

source ~/Projects/my_env/bin/activate

python /home/mila/s/soroush.omranpour/scratch/OmniVLMPlus/pretrain.py --token_reduction="$1" --reduction_factor="$2" --image_pe="$3" --lr_proj="$4" --freeze_lm --freeze_enc
