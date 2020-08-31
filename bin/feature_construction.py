#!/usr/bin/env python

import argparse
import collections
import math
from time import localtime, strftime
import typing

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

def flatten(l): return [item for sublist in l for item in sublist]
def get_mf(graph, nodepairs, residual, flow_func, **kwargs): 
  return [nx.maximum_flow_value(graph, *nodepair, capacity='weight', flow_func=flow_func, residual=residual, **kwargs) for nodepair in nodepairs]
def get_katz(graph, nodepairs, beta=.005, cutoff=5): 
  return [sum([beta**k * v for k, v in collections.Counter([len(p) for p in nx.all_simple_paths(graph, *nodepair, cutoff=5)]).items()]) for nodepair in nodepairs]
def get_propflow(graph, limit=5):
    score = dict()
    for node in tqdm(graph):
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
def print_status(desc: str): print(f'{strftime("%H:%M:%S", localtime())}: {desc}')
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

        
def feature_construction(graph: nx.Graph, nodepairs: typing.List[typing.Tuple[int, int]], target: typing.List[bool], hplp=True, chunksize=1000, n_jobs=256) -> pd.DataFrame:
  features = dict()
  
  # Single-core calculations:
  ## Degree
  degree = np.array([[degree for _, degree in graph.degree(nodepair)] for nodepair in tqdm(nodepairs, desc="Degree")])
  degree.sort(axis=1)
  features['d_min'] = degree[:,0]
  features['d_max'] = degree[:,1]
  
  ## Volume
  volume = np.array([[degree for _, degree in graph.degree(nodepair, weight='weight')] for nodepair in tqdm(nodepairs, desc="Volume")])
  volume.sort(axis=1)
  features['v_min'] = volume[:,0]
  features['v_max'] = volume[:,1]
  
  ## Common Neighbors
  features['cn'] = np.array([len(list(nx.common_neighbors(graph, *nodepair))) for nodepair in tqdm(nodepairs, desc='Common Neighbors')])
  
  ## Propflow
  print_status("Calculate propflow.")
  score = get_propflow(graph)
  features['pf'] = np.fromiter(((score.get(u, 0).get(v, 0) + score.get(v, 0).get(u, 0))/2 for u, v in tqdm(nodepairs, desc='propflow')), dtype=float)
  
  ## Shortest Paths
  sp_dict = {node: {k: len(v) for k, v in nx.predecessor(graph, node, cutoff=5).items()} for node in tqdm(graph, desc='calculating shortest paths')}
  
  print_status("Store shortest paths.")
  sp = np.fromiter((sp_dict[u][v] for u, v in tqdm(nodepairs, desc='Retrieve sp')), dtype=int)
  features['sp'] = sp
  
  ## Adamic Adar
  if not hplp: features['aa'] = [sum([s for _, _, s in nx.adamic_adar_index(graph, [nodepair, tuple(reversed(nodepair))])]) / 2 for nodepair in tqdm(nodepairs, desc='Adamic Adar')]
  
  ## Jaccard Coefficient
  if not hplp: features['jc'] = [p for _, _, p in nx.jaccard_coefficient(graph, tqdm(nodepairs, desc='Jaccard Coefficient'))]
  
  ## Preferential Attachment
  if not hplp: features['pa'] = [p for _, _, p in nx.preferential_attachment(graph, tqdm(nodepairs, desc='Preferential Attachment'))]
    
  joblib.dump(features, args.directory + 'singlecore.pkl', protocol=5)
  
  # Multi-core calculations:
  no_chunks = len(nodepairs) // chunksize
  nodepair_chuncks = np.array_split(nodepairs, no_chunks)
  
  ## Maxflow
  residual = nx.algorithms.flow.utils.build_residual_network(graph, 'weight')

  kwargs = {'flow_func': nx.algorithms.flow.preflow_push} if args.preflow else {'flow_func': nx.algorithms.flow.edmonds_karp, 'cutoff': 5}

  mf = np.array(
    flatten(ProgressParallel(n_jobs=n_jobs, total=no_chunks, desc='Maxflow (parallel)')(joblib.delayed(get_mf)(graph, nodepair_chunck, residual, **kwargs) for nodepair_chunck in nodepair_chuncks))
  )
  print_status("Store maxflow.")
  mf = np.array(mf)
  mf.dump(args.directory + 'maxflow.pkl')
  features['mf'] = mf
  
  # Katz
  if not hplp:
    katz = np.array(flatten(ProgressParallel(n_jobs=n_jobs, total=no_chunks, desc='Katz (parallel)')(joblib.delayed(get_katz)(graph, nodepair_chunck) for nodepair_chunck in nodepair_chuncks)))
    print_status("Store Katz.")
    katz.dump(args.directory + 'katz.pkl')
    features['katz'] = katz
    
  # Target
  features['target'] = target
  
  return pd.DataFrame(features)
if __name__ == "__main__":

  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', help='Location where distances.pkl, graph.pkl, nodepairs.pkl and targets.pkl are present. Result is stored as features.pkl in this directory.')
  parser.add_argument('--hplp', help='Only evaluate HPLP features.', action='store_true')
  parser.add_argument('--skipmaxflow', help='Skip maxflow feature.', action='store_true')
  parser.add_argument('--preflow', help='Use preflow push algorithm instead of Edmonds_karp.', action='store_true')
  args = parser.parse_args()

  print_status('Load nodepairs')
  nodepairs = joblib.load(args.directory + 'nodepairs.pkl')

  print_status('Start collecting features')
  features = feature_construction(
    graph = joblib.load(args.directory + 'graph.pkl'),
    nodepairs = joblib.load(args.directory + 'nodepairs.pkl'),
    target = joblib.load(args.directory + 'target.pkl'),
    hplp=args.hplp
  )
  
  # Store
  print_status("Store features.") 
  features.to_pickle(args.directory + 'features.pkl', protocol=5)

  