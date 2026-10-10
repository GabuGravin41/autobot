
import os, re, glob, time, pickle, math
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
T0 = time.time()

def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not hits: raise FileNotFoundError(name)
    return sorted(hits, key=len)[0]

COMP   = os.path.dirname(find('test.parquet'))
TRAIN  = os.path.join(COMP, 'train.parquet')
TEST   = os.path.join(COMP, 'test.parquet')
SAMPLE = os.path.join(COMP, 'sample_submission.csv')
print('competition files:', os.listdir(COMP))
