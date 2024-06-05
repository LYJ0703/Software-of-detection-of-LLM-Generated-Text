#!/bin/bash
#SBATCH -p gpu3              # Request the GPU partition
#SBATCH -N 1                 # Request 1 node
#SBATCH -c 1                 # Request 1 CPU core
#SBATCH --mem 18000          # Request 10GB of memory
#SBATCH --gres gpu:2         # Request 1 GPU

echo "Bscore begin"
python3 Bscore.py -t "$1"