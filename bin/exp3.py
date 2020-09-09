#!/usr/bin/env python

import os
from time import localtime, strftime

import joblib

from linkprediction import construct_edges, convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component
from feature_construction import feature_construction, store

def print_status(*args): print(strftime("%H:%M:%S", localtime()), *args)

# # AU
# store(edges = read_edges('data/au.edges', sep='\t'), filepath = 'exp3/au/', directed=True)

# # Condmat
# store(edges = construct_edges('data/condmat.hg2'), filepath = 'exp3/condmat/', directed=False)

# # Digg
# store(edges = read_edges('data/digg.edges'), filepath = 'exp3/digg/', directed=True)

# # Enron
# store(edges = read_edges('data/enron.edges'), filepath = 'exp3/enron/', directed=True)

# # Slashdot
# store(edges = read_edges('data/slashdot.edges', skiprows=2), filepath = 'exp3/slashdot/', directed=True)

# # SO
# store(edges = read_edges('data/so.edges', sep='\t'), filepath = 'exp3/so/', directed=True)

# # Feature construction
# joblib.Parallel(n_jobs=18)(
#   joblib.delayed(feature_construction)(path, n_jobs=1, position=index, only_singlecore=True) 
#   for index, path in enumerate([f'exp3/{graph}/{type_test}/' for graph in ['au', 'condmat', 'digg', 'enron', 'slashdot', 'so'] for type_test in ['random', 'train', 'test']])
# )

for graph in ['au', 'condmat', 'digg', 'enron', 'slashdot', 'so']:
  for type_test in ['random', 'train', 'test']:
    print_status(graph, type_test)
    feature_construction(path=f'exp3/{graph}/{type_test}/', use_tqdm=True)

