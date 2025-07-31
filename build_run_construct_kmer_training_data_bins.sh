#!/bin/bash

for i in {1..22}
do
    bash run_construct_kmer_training_data_bins.sh ${i} > slurm/run_construct_kmer_training_data_bins_chr${i}.sh
    sbatch slurm/run_construct_kmer_training_data_bins_chr${i}.sh
done