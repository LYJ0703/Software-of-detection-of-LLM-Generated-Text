#!/bin/bash
#SBATCH -p gpu3              # Request the GPU partition
#SBATCH -N 1                 # Request 1 node
#SBATCH -c 1                 # Request 1 CPU core
#SBATCH --mem 8000          # Request 10GB of memory
#SBATCH --gres gpu:1         # Request 1 GPU

echo "ppl begin"
python3 ppl.py -t "$1"