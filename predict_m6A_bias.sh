#!/bin/bash
#SBATCH --job-name=predict_m6A_bias
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=500g
#SBATCH --time=7-00:00:00
#SBATCH --account=apboyle99
#SBATCH --partition=largemem
#SBATCH --output=logs/%x.log

source ~/.bashrc
conda init
conda activate cnn

python predict_m6A_bias.py

exit