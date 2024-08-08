#!/bin/bash
#SBATCH --job-name=rer_wer
#SBATCH --output=job_%A.out
#SBATCH --error=job_%A.err
#SBATCH --mem=80Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --time=18:00:00
#SBATCH -x node[113,115]

srun python main.py
