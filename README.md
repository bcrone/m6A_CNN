# m6A_CNN
m6A methylation bias correction via CNN

1) Install conda CNN environment

   conda create -f environment.yml


2) Build kmer training data per-chromosome (sequence, probability bin, count)

   Adjust parameters as needed (flanking window, bin thresholds, etc.)

   bash build_run_construct_kmer_training_data.sh


3) Merge per-chromosome kmer parquets into single parquet

   sbatch merge_kmer_parquets_binned.sh


4) Train model

   sbatch predict_m6A_bias.sh
    
