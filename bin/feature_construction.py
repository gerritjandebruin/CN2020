#!/usr/bin/env python

import argparse
import collections
import math
import os
from time import localtime, strftime

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from linkprediction import construct_edges, convert_to_set, get_distances, read_edges, filter_edges, get_graph, giant_component

def flatten(l): return [item for sublist in l for item in sublist]
def get_mf(graph, nodepairs, residual, flow_func, **kwargs): 
  if type(graph) is nx.DiGraph:
    return [nx.maximum_flow_value(graph, *nodepair, capacity='weight', flow_func=flow_func, residual=residual, **kwargs) for nodepair in nodepairs]
  else:
    return [
      (
        nx.maximum_flow_value(graph, u, v, capacity='weight', flow_func=flow_func, residual=residual, **kwargs) + 
        nx.maximum_flow_value(graph, v, u, capacity='weight', flow_func=flow_func, residual=residual, **kwargs)
      ) / 2
      for u, v in nodepairs
    ]
def get_katz(graph, nodepairs, beta=.005, cutoff=5): 
  return [sum([beta**k * v for k, v in collections.Counter([len(p) for p in nx.all_simple_paths(graph, *nodepair, cutoff=5)]).items()]) for nodepair in nodepairs]
def get_propflow(graph, limit=5):
    score = dict()
    for node in graph:
        scores = {node: 1.0}
        found = set()
        newSearch = [node]

        for _ in range(0, limit+1):
            oldSearch = list(newSearch)
            found.update(newSearch)
            newSearch = set()
            while len(oldSearch) != 0:
                n2 = oldSearch.pop()
                nodeInput = scores[n2]
                degree = graph.degree(n2, 'weight')
                flow = 0.0
                for n3 in graph[n2]:
                    wij = graph[n2][n3]['weight']
                    flow = nodeInput * (wij*1.0/degree)
                    scores[n3] = scores.get(n3, 0) + flow
                    if n3 not in found: newSearch.add(n3)
        score[node] = scores
    return score  
def print_status(desc: str, index: int): print(f'{strftime("%H:%M:%S", localtime())} {desc}') if index is None else print(f'{strftime("%H:%M:%S", localtime())} {index} {desc}')
class ProgressParallel(joblib.Parallel):
    def __init__(self, use_tqdm=True, total=None, desc=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc=self._desc) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

        
def feature_construction(path: str, *, preflow=False, chunksize=500, n_jobs=256, position=None, use_tqdm=False, only_singlecore=False) -> pd.DataFrame:
  print_status('Start loading', position)
  graph     = joblib.load(path + 'graph.pkl')
  nodepairs = joblib.load(path + 'nodepairs.pkl')
  target    = joblib.load(path + 'targets.pkl')
  
  if os.path.isfile(path + 'singlecore.pkl'): 
    print_status('Load singlecore.pkl', position)
    features = joblib.load(path + 'singlecore.pkl')
  else:
    features = dict()

    # Degree
    print_status('degree', position)
    if type(graph) is nx.DiGraph:
      features['d_out_u'] = [graph.out_degree(u) for u, _ in nodepairs]
      features['d_out_v'] = [graph.out_degree(v) for _, v in nodepairs]
      features['d_in_u'] = [graph.in_degree(u) for u, _ in nodepairs]
      features['d_in_v'] = [graph.in_degree(v) for _, v in nodepairs]
    else:
      degree = np.array([[degree for _, degree in graph.degree(nodepair)] for nodepair in nodepairs])
      degree.sort(axis=1)
      features['d_min'] = degree[:,0]
      features['d_max'] = degree[:,1]
    
    # Volume
    print_status('volume', position)
    if type(graph) is nx.DiGraph:
      features['v_out_u'] = [graph.out_degree(u, weight='weight') for u, _ in nodepairs]
      features['v_out_v'] = [graph.out_degree(v, weight='weight') for _, v in nodepairs]
      features['v_in_u'] = [graph.in_degree(u, weight='weight') for u, _ in nodepairs]
      features['v_in_v'] = [graph.in_degree(v, weight='weight') for _, v in nodepairs]
    else:
      volume = np.array([[degree for _, degree in graph.degree(nodepair, weight='weight')] for nodepair in nodepairs])
      volume.sort(axis=1)
      features['v_min'] = volume[:,0]
      features['v_max'] = volume[:,1]
    
    # Common Neighbors
    print_status('cn', position)
    features['cn'] = [len(set(graph.neighbors(u)) & set(graph.neighbors(v))) for u, v in nodepairs] if type(graph) is nx.DiGraph else [len(list(nx.common_neighbors(graph, *np))) for np in nodepairs]
    
    # Propflow
    print_status('pf', position)
    score = get_propflow(graph)
    if type(graph) is nx.DiGraph: features['pf'] = np.fromiter((score.get(u, 0).get(v, 0) for u, v in nodepairs), dtype=float)
    else:  features['pf'] = np.fromiter(((score.get(u, 0).get(v, 0) + score.get(v, 0).get(u, 0))/2 for u, v in nodepairs), dtype=float)
    
    # Shortest Paths
    print_status('sp', position)
    sp_dict = {node: {k: len(v) for k, v in nx.predecessor(graph, node, cutoff=5).items()} for node in graph}
    
    sp = np.fromiter((sp_dict[u][v] for u, v in nodepairs), dtype=int)
    features['sp'] = sp
    
  #   # Adamic Adar
  #   features['aa'] = [sum([s for _, _, s in nx.adamic_adar_index(graph, [nodepair, tuple(reversed(nodepair))])]) / 2 for nodepair in tqdm(nodepairs, desc='Adamic Adar')]
    
  #   # Jaccard Coefficient
  #   features['jc'] = [p for _, _, p in nx.jaccard_coefficient(graph, tqdm(nodepairs, desc='Jaccard Coefficient'))]
    
  #   # Preferential Attachment
  #   features['pa'] = [p for _, _, p in nx.preferential_attachment(graph, tqdm(nodepairs, desc='Preferential Attachment'))]
    
    # Target
    features['target'] = target
    
    print_status('store singlecore.pkl', position)
    joblib.dump(features, path + 'singlecore.pkl', protocol=5)
    
  if only_singlecore: return

  # Multi-core calculations:
  no_chunks = len(nodepairs) // chunksize
  nodepair_chuncks = np.array_split(nodepairs, no_chunks)

  ## Maxflow
  print_status('mf', position)
  residual = nx.algorithms.flow.utils.build_residual_network(graph, 'weight')

  kwargs = dict(flow_func=nx.algorithms.flow.preflow_push) if preflow else dict(flow_func=nx.algorithms.flow.edmonds_karp, cutoff=5)

  features['mf'] = np.array(flatten(ProgressParallel(n_jobs=n_jobs, desc='mf', total=len(nodepair_chuncks), use_tqdm=use_tqdm)(
    joblib.delayed(get_mf)(graph, nodepair_chunck, residual, **kwargs) for nodepair_chunck in nodepair_chuncks))
  )

#   # Katz
#   if not hplp:
#     katz = np.array(flatten(ProgressParallel(n_jobs=n_jobs, total=no_chunks, desc='Katz (parallel)')(joblib.delayed(get_katz)(graph, nodepair_chunck) for nodepair_chunck in nodepair_chuncks)))
#     print_status("Store Katz.")
#     katz.dump(path + 'katz.pkl')
#     features['katz'] = katz

  print_status('store', position)
  pd.DataFrame(features).to_pickle(path + 'features.pkl', protocol=5)
  
def random(edges: list, *, directed: bool, t_a=50000, t_b=70000, cutoff=2, verbose=False, **kwargs):
  if verbose: print_status('random')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  print(graph.number_of_nodes())
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff, **kwargs)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def train(edges: list, *, directed: bool, t_a=50000, t_b=60000, cutoff=2, verbose=False, **kwargs):
  if verbose: print_status('train')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff, **kwargs)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def test(edges: list, *, directed: bool, t_a=60000, t_b=70000, cutoff=2, verbose=False, **kwargs):
  if verbose: print_status('test')
  edges_mature = filter_edges(edges, stop=t_a)
  edges_probe = filter_edges(edges, start=t_a, stop=t_b)
  graph = giant_component(get_graph(edges_mature, directed=directed))
  uv_probes = convert_to_set(edges_probe)
  nodepairs, _ = get_distances(graph, cutoff=cutoff, **kwargs)
  targets = [nodepair in uv_probes for nodepair in nodepairs]
  return dict(nodepairs=nodepairs, graph=graph, targets=targets)

def store(edges, filepath, *, directed: bool, verbose=False, **kwargs):
  if verbose: print_status(filepath)
  for path in ['random/', 'train/', 'test/']: os.makedirs(filepath + path, exist_ok=True)
  for name, obj in random(edges, directed=directed, verbose=verbose, **kwargs).items(): joblib.dump(obj, f'{filepath}random/{name}.pkl', protocol=5)
  for name, obj in train(edges, directed=directed, verbose=verbose, position=position, **kwargs).items(): joblib.dump(obj, f'{filepath}train/{name}.pkl', protocol=5)
  for name, obj in test(edges, directed=directed, verbose=verbose, position=position, **kwargs).items(): joblib.dump(obj, f'{filepath}test/{name}.pkl', protocol=5)

if __name__ == "__main__":
  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', help='Location where distances.pkl, graph.pkl, nodepairs.pkl and targets.pkl are present. Result is stored as features.pkl in this directory.')
  args = parser.parse_args()

  features = feature_construction(path = args.directory, use_tqdm=True)