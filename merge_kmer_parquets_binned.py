import numpy as np
import pandas as pd
import fastparquet

for i in reversed(range(1,22)):
    chrAll = pd.read_parquet(f'/path/to/binned_chr{i+1}.summary.parquet.gzip')
    chrom = pd.read_parquet(f'/path/to/binned_chr{i}.parquet.gzip')
    merged = pd.merge(chrAll, chrom, on=['seq','prob'], how='outer', suffixes=('_1','_2'))
    merged['count'] = merged[['count_1', 'count_2']].fillna(0).sum(axis=1).astype(int)
    merged.drop(columns=['count_1','count_2'],inplace=True)
    merged.to_parquet(f'/path/to/binned_chr{i}.summary.parquet.gzip', compression='gzip')
    del chrAll
    del chrom
    del merged