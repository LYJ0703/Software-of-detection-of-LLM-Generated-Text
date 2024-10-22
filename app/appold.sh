#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 30000
#SBATCH --gres gpu:1

echo "job begin"
python appold.py
echo "job end"

