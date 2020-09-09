#!/usr/bin/env python

import os
from time import localtime, strftime

import joblib

from linkprediction import construct_edges, convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component
from feature_construction import feature_construction

def print_status(*args): print(strftime("%H:%M:%S", localtime()), *args)

def random(edges: list, *, directed: bool, t_a=50000, t_b=70000, cutoff=2):
  print('random')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  print(graph.number_of_nodes())
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def train(edges: list, *, directed: bool, t_a=50000, t_b=60000, cutoff=2):
  print('train')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def test(edges: list, *, directed: bool, t_a=60000, t_b=70000, cutoff=2):
  print('test')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def store(edges, filepath, *, directed: bool):
  print(filepath)
  for path in ['random/', 'train/', 'test/']: os.makedirs(filepath + path, exist_ok=True)
  for name, obj in random(edges, directed=directed).items(): joblib.dump(obj, f'{filepath}random/{name}.pkl', protocol=5)
  for name, obj in train(edges, directed=directed).items(): joblib.dump(obj, f'{filepath}train/{name}.pkl', protocol=5)
  for name, obj in test(edges, directed=directed).items(): joblib.dump(obj, f'{filepath}test/{name}.pkl', protocol=5)

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

