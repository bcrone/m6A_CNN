#!/bin/bash
cat <<EOF
#!/bin/bash
#SBATCH --job-name=run_construct_kmer_training_data_bins_chr${1}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100g
#SBATCH --time=15:00:00
#SBATCH --account=
#SBATCH --partition=
#SBATCH --output=logs/%x.log

source ~/.bashrc
conda init
conda activate cnn

C="$1"
BAM="/path/to/hg38.analysisSet.aligned.filtered.sorted.bam"
CHR="chr\${C}"
PREFIX=""
WINDOW="6"

python /path/to/construct_kmer_training_data_bins.py -b \${BAM} -c \${CHR} -p \${PREFIX} -w \${WINDOW}

exit
EOF