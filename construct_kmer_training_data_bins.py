import argparse
from collections import defaultdict
import fastparquet
import numpy as np 
import pandas as pd
import pysam

def reverse_complement(sequence):
    complement = {"A":"T", "T":"A", "C":"G", "G":"C"}
    r_complement = ''.join(complement[base] for base in reversed(sequence))
    return r_complement

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bam")
parser.add_argument("-c", "--chrom")
parser.add_argument("-p", "--prefix")
parser.add_argument("-w", "--window")
args = parser.parse_args()

window = int(args.window)
out_parquet = f'{args.prefix}_{args.chrom}.parquet.gzip'
kmer_probs = defaultdict(list)

bam = pysam.AlignmentFile(args.bam)
count = 0
print(f'Reading {args.bam} for chrom {args.chrom}')
for read in bam.fetch(args.chrom):
    count += 1
    print(count)
    if read.has_tag('MM'):
        if read.is_forward:
            indicies = [i for i, ltr in enumerate(read.query_sequence) if ltr == "A"]
        else:
            indicies = [i for i, ltr in enumerate(reverse_complement(read.query_sequence)) if ltr =="A"]
        mm_tags = read.get_tag('MM').split(';')
        ml_probs = list(map(lambda x: x / 256, list(read.get_tag('ML'))))
        for tag in mm_tags:
            if tag.startswith('A+a'):
                m6A_pos = []
                pos = -1
                m6A_tags = list(map(int,tag.split(',')[1:]))
                for i in range(len(m6A_tags)):
                    if m6A_tags[i] == 0:
                        pos += 1
                    else:
                        pos += m6A_tags[i] + 1
                    m6A_pos.append(pos)
        
        m6A_indicies = [indicies[i] for i in m6A_pos]
        m6A_probs = ml_probs[:len(m6A_indicies)]
        l_seq_bound = window
        u_seq_bound = len(read.query_sequence) - window
        
        for i in range(len(m6A_indicies)):
            m6A_index = m6A_indicies[i]
            if m6A_index < l_seq_bound or m6A_index > u_seq_bound:
                continue
            if read.is_forward:
                kmer = read.query_sequence[m6A_index - window : m6A_index + window + 1]
            else:
                kmer = reverse_complement(read.query_sequence)[m6A_index - window :  m6A_index + window + 1]
            kmer_probs[kmer].append(m6A_probs[i])

bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
kmer_out = {}
for k,v in kmer_probs.items():
    map = np.digitize(v, bins, right=False)
    prob, counts = np.unique(map, return_counts = True)
    result = dict(zip(prob,counts))
    for p,c in result.items():
        kmer_out[(k,p)] = c

s = pd.Series(kmer_out)
kmer_probs_table = s.reset_index()
kmer_probs_table.columns = ['seq','prob','count']
kmer_probs_table.to_parquet(out_parquet,compression='gzip')