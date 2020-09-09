#!/usr/bin/env python

from itertools import product
import os
from time import localtime, strftime

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from linkprediction import read_edges, store
from feature_construction import feature_construction

def print_status(*args): print(strftime("%H:%M:%S", localtime()), *args)
  
# Set-up
# edges = read_edges('data/au.edges', sep='\t')
# for i in range(1, 11): store(edges, f'exp4/{i}/', directed=True, t_a=70000*i) 
# Parallel(n_jobs=30)(delayed(store)(edges, f'exp4/{i}/', directed=True, t_a=70000*i, only_do=do, position=pos) for pos, (i, do) in enumerate(product(range(1, 11), ['random', 'train', 'test'])))

# Feature construction
Parallel(n_jobs=9)(
  delayed(feature_construction)(f'exp4/{i}/{do}/', position=position, only_singlecore=True, verbose=True) for position, (i, do) in enumerate(product(range(4, 11), ['random', 'train', 'test']))
)

for i in range(1, 11):
  print_status(graph, type_test)
  feature_construction(path=f'exp4/{i}/', use_tqdm=True)