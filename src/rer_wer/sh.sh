#!/bin/bash
#SBATCH --job-name=rer_wer
#SBATCH --output=job_%A.out
#SBATCH --error=job_%A.err
#SBATCH --mem=40Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=18:00:00
#SBATCH -x node[113,115]

source activate /om2/user/fabiocat/myconda/envs/rer_wer

srun python main.py
