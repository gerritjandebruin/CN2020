#!/usr/bin/env python

import os
from time import localtime, strftime

import joblib
import numpy as np
import pandas as pd

from linkprediction import construct_edges, convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component

def print_status(*args): print(strftime("%H:%M:%S", localtime()), *args)
  
def random(edges: list, *, directed: bool, t_a=50000, t_b=None, cutoff=2):
  if t_b is None: t_b = t_a + 20000
  print('random')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  print(graph.number_of_nodes())
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def train(edges: list, *, directed: bool, t_a=50000, t_b=None, cutoff=2):
  if t_b is None: t_b = t_a + 10000
  print('train')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def test(edges: list, *, directed: bool, t_a=60000, t_b=None, cutoff=2):
  if t_b is None: t_b = t_a + 10000
  print('test')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def store(edges, filepath, *, directed: bool, **kwargs):
  print_status(filepath)
  for path in ['random/', 'train/', 'test/']: os.makedirs(filepath + path, exist_ok=True)
  for name, obj in random(edges, directed=directed, **kwargs).items(): joblib.dump(obj, f'{filepath}random/{name}.pkl', protocol=5)
  for name, obj in train(edges, directed=directed, **kwargs).items(): joblib.dump(obj, f'{filepath}train/{name}.pkl', protocol=5)
  for name, obj in test(edges, directed=directed, **kwargs).items(): joblib.dump(obj, f'{filepath}test/{name}.pkl', protocol=5)
  
# Set-up
edges = read_edges('data/au.edges', sep='\t')
for i in range(1, 11): store(edges, f'exp4/{i}/', directed=True, t_a=70000*i) 

# Feature construction
joblib.Parallel(n_jobs=30)(joblib.delayed(feature_construction)(path, n_jobs=1, position=index, only_singlecore=True) for index, path in enumerate([f'exp4/{i}/' for i in range(1, 11)]))

for i in range(1, 11):
  print_status(graph, type_test)
  feature_construction(path=f'exp4/{i}/', use_tqdm=True)