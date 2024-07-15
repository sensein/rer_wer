#!/bin/bash
#SBATCH --job-name=rer_wer
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH --mem=80Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=gablab

source activate /om2/user/fabiocat/myconda/envs/rer_wer

srun python main.py
