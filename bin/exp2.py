#!/usr/bin/env python

from itertools import product
import os
from time import localtime, strftime

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from linkprediction import construct_edges, convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component
from feature_construction import store

def print_status(*args): print(strftime("%H:%M:%S", localtime()), *args)
  
# Set-up
edges = read_edges('data/au.edges', sep='\t')
# for i in range(1, 11): store(edges, f'exp4/{i}/', directed=True, t_a=70000*i) 
Parallel(n_jobs=30)(delayed(store)(edges, f'exp4/{i}/', directed=True, t_a=70000*i, only_do=do, position=pos) for pos, (i, do) in enumerate(product(range(1, 11), ['random', 'train', 'test'])))

# Feature construction
Parallel(n_jobs=30)(delayed(feature_construction)(path, n_jobs=1, position=index, only_singlecore=True) for index, path in enumerate([f'exp4/{i}/' for i in range(1, 11)]))

for i in range(1, 11):
  print_status(graph, type_test)
  feature_construction(path=f'exp4/{i}/', use_tqdm=True)