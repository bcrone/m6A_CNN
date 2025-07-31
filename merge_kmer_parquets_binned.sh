#!/bin/bash
#SBATCH --job-name=merge_kmer_parquets_binned
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=200g
#SBATCH --time=7-00:00:00
#SBATCH --account=apboyle99
#SBATCH --partition=largemem
#SBATCH --output=logs/%x.log

source ~/.bashrc
conda init
conda activate cnn

python merge_kmer_parquets_binned.py

exit