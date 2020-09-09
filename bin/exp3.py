#!/usr/bin/env python

import datetime
from itertools import combinations
import os

import joblib
import pandas as pd

from linkprediction import convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component
from feature_construction import feature_construction

# def random(edges: list, *, directed: bool, t_a=50000, t_b=70000, cutoff=2):
#   edges_mature = filter_edges(edges, stop=t_a)
#   edges_probe = filter_edges(edges, start=t_a, stop=t_b)
#   graph = giant_component(get_graph(edges_mature, directed=directed))
#   print(graph.number_of_nodes())
#   uv_probes = convert_to_set(edges_probe)
#   nodepairs, _ = get_distances(graph, cutoff=cutoff)
#   targets = [nodepair in uv_probes for nodepair in nodepairs]
#   return dict(nodepairs=nodepairs, graph=graph, targets=targets)

# def train(edges: list, *, directed: bool, t_a=50000, t_b=60000, cutoff=2):
#   edges_mature = filter_edges(edges, stop=t_a)
#   edges_probe = filter_edges(edges, start=t_a, stop=t_b)
#   graph = giant_component(get_graph(edges_mature, directed=directed))
#   uv_probes = convert_to_set(edges_probe)
#   nodepairs, _ = get_distances(graph, cutoff=cutoff)
#   targets = [nodepair in uv_probes for nodepair in nodepairs]
#   return dict(nodepairs=nodepairs, graph=graph, targets=targets)

# def test(edges: list, *, directed: bool, t_a=60000, t_b=70000, cutoff=2):
#   edges_mature = filter_edges(edges, stop=t_a)
#   edges_probe = filter_edges(edges, start=t_a, stop=t_b)
#   graph = giant_component(get_graph(edges_mature, directed=directed))
#   uv_probes = convert_to_set(edges_probe)
#   nodepairs, _ = get_distances(graph, cutoff=cutoff)
#   targets = [nodepair in uv_probes for nodepair in nodepairs]
#   return dict(nodepairs=nodepairs, graph=graph, targets=targets)

# def store(edges, filepath, *, directed: bool):
#   print(filepath)
#   for path in ['random/', 'train/', 'test/']: os.makedirs(filepath + path, exist_ok=True)
#   for name, obj in random(edges, directed=directed).items(): joblib.dump(obj, f'exp3/au/random/{name}.pkl', protocol=5)
#   for name, obj in train(edges, directed=directed).items(): joblib.dump(obj, f'exp3/au/train/{name}.pkl', protocol=5)
#   for name, obj in test(edges, directed=directed).items(): joblib.dump(obj, f'exp3/au/test/{name}.pkl', protocol=5)

# # AU
# # store(edges = read_edges('data/au.edges', sep='\t'), filepath = 'exp3/graphs/au/', directed=True)

# # Condmat
# def get_papers(file):
#   papers = list()
#   # Get number of rows to read for the vertices.
#   with open(file) as f: no_rows = int(f.readline().split(' ')[1])
 
#   with open(file) as f:
#     for paper in f.readlines()[no_rows+2:]:
#       # Each line has the following format: epoch no_authors [ u v (w ...) ]
#       epoch = datetime.datetime.fromtimestamp(int(paper.split(' ')[0]))
          
#       no_authors = int(paper.split(' ')[1])
#       index1 = paper.find('[')+2
#       index2 = paper.find(']')-1

#       authors = [int(auth) for auth in paper[index1:index2].split(' ')]
#       assert no_authors == len(authors)
      
#       papers.append((authors, epoch))
#   return papers 
# def get_edgelist(file='data/condmat.hg2'): 
#   return pd.DataFrame([(u, v, date) if u<v else (v, u, date) for authors, date in get_papers(file) for u, v in combinations(authors, 2)], columns=['source', 'target', 'date'])
# store(edges = get_edgelist(), filepath = 'exp3/graphs/condmat/', directed=False)

# # Digg
# store(edges = read_edges('data/digg.edges'), filepath = 'exp3/graphs/digg/', directed=True)

# # Enron
# store(edges = read_edges('data/enron.edges'), filepath = 'exp3/graphs/enron/', directed=True)

# # Slashdot
# store(edges = read_edges('data/slashdot.edges', skiprows=2), filepath = 'exp3/graphs/slashdot/', directed=True)

# # SO
# store(edges = read_edges('data/so.edges', sep='\t'), filepath = 'exp3/graphs/so/', directed=True)

# Feature construction
joblib.Parallel(n_jobs=18)(
  joblib.delayed(feature_construction)(path, n_jobs=1, position=index, only_singlecore=True) 
  for index, path in enumerate([f'exp3/{graph}/{type_test}/' for graph in ['au', 'condamat', 'digg', 'enron', 'slashdot', 'so'] for type_test in ['random', 'train', 'test']])
)

