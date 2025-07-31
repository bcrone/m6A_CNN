#!/bin/bash
#SBATCH --job-name=validate_m6A_bias
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=200g
#SBATCH --time=1-00:00:00
#SBATCH --account=apboyle99
#SBATCH --partition=largemem
#SBATCH --output=logs/%x.log

source ~/.bashrc
conda init
conda activate cnn

python validate_m6A_bias.py

exit