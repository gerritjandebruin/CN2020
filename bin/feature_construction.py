#!/usr/bin/env python

import argparse
import collections
import math
from time import localtime, strftime

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

def flatten(l): return [item for sublist in l for item in sublist]
def get_maxflow(graph, nodepairs, residual): return [nx.maximum_flow_value(graph, *nodepair, capacity='weight', flow_func=nx.algorithms.flow.preflow_push, residual=residual) for nodepair in nodepairs]
def get_shortest_paths(graph, nodepairs): return [len(list(nx.all_shortest_paths(graph, *nodepair))) for nodepair in nodepairs]
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

if __name__ == "__main__":
  chunk_size = 1000

  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', help='Location where distances.pkl, graph.pkl, nodepairs.pkl and targets.pkl are present. Result is stored as features.pkl in this directory.')
  args = parser.parse_args()
  
  # Read graph and nodepairs
  print_status('Load graph')
  graph = joblib.load(args.directory + 'graph.pkl')

  print_status('Load nodepairs')
  nodepairs = joblib.load(args.directory + 'nodepairs.pkl')

  print_status('Load target')
  target = joblib.load(args.directory + 'targets.pkl')


  print_status('Start collecting features')
  
  # Single-core calculations:
  ## Degree
  degree_min, degree_max = zip(*[sorted([degree for _, degree in graph.degree(nodepair)]) for nodepair in tqdm(nodepairs, desc="degree")])
  
  ## Volume
  volume_min, volume_max = zip(*[sorted([degree for _, degree in graph.degree(nodepair, weight='weight')]) for nodepair in tqdm(nodepairs, desc="volume")])
  
  ## Common nbrs
  common_nbrs = [len(list(nx.common_neighbors(graph, *nodepair))) for nodepair in tqdm(nodepairs, desc='Common nbrs')]
  
  ## Adamic Adar
  adamic_adar = [sum([s for _, _, s in nx.adamic_adar_index(graph, [nodepair, tuple(reversed(nodepair))])]) / 2 for nodepair in tqdm(nodepairs, desc='Adamic Adar')]
  
  ## Jaccard
  jaccard = [p for _, _, p in nx.jaccard_coefficient(graph, tqdm(nodepairs, desc='jaccard'))]
  
  ## Preferential Attachment
  preferential_attachment = [p for _, _, p in nx.preferential_attachment(graph, tqdm(nodepairs, desc='preferential attachment'))]
  
  # Store
  print_status('Start storing features')
  pd.DataFrame(
    dict(
      degree_min=degree_min, degree_max=degree_max, volume_min=volume_min, volume_max=volume_max,common_nbrs=common_nbrs,adamic_adar=adamic_adar,jaccard=jaccard, 
      preferential_attachment=preferential_attachment
    )
  ).to_pickle(args.directory + 'singlecore.pkl')

  ## Propflow
  print_status("Calculate propflow.")
  score = get_propflow(graph)
  propflow = [(score.get(u, 0).get(v, 0) + score.get(v, 0).get(u, 0))/2 for u, v in tqdm(nodepairs, desc='propflow')]
  joblib.dump(propflow, args.directory + 'propflow.pkl') 
  
  # Multi-core calculations:
  no_chunks = len(nodepairs) // chunk_size
  nodepair_chuncks = np.array_split(nodepairs, no_chunks)
  
  ## Maxflow
  print_status("Build residual network.")
  residual = nx.algorithms.flow.utils.build_residual_network(graph, 'weight')
  
  maxflow = flatten(ProgressParallel(n_jobs=-1, total=no_chunks, desc='Maxflow (parallel)')(joblib.delayed(get_maxflow)(graph, nodepair_chunck, residual) for nodepair_chunck in nodepair_chuncks))
  print_status("Store maxflow.")
  joblib.dump(maxflow, args.directory + 'maxflow.pkl')
  
  ## Shortest Paths
  shortest_paths = flatten(
    ProgressParallel(n_jobs=-1, total=no_chunks, desc='Shortest Paths (parallel)')(joblib.delayed(get_shortest_paths)(graph, nodepair_chunck) for nodepair_chunck in nodepair_chuncks)
  )
  print_status("Store shortest paths.")
  joblib.dump(shortest_paths, args.directory + 'shortest_paths.pkl')
  
  ## Katz
  katz = flatten(ProgressParallel(n_jobs=-1, total=no_chunks, desc='Katz (parallel)')(joblib.delayed(get_katz)(graph, nodepair_chunck) for nodepair_chunck in nodepair_chuncks))
  print_status("Store Katz.")
  joblib.dump(katz, args.directory + 'katz.pkl')
  
  print_status("Construct dataframe")  
  features = dict(
    degree_min=degree_min, 
    degree_max=degree_max, 
    volume_min=volume_min, 
    volume_max=volume_max,
    common_nbrs=common_nbrs,
    adamic_adar=adamic_adar,
    jaccard=jaccard,
    preferential_attachment=preferential_attachment,
    maxflow=maxflow,
    shortest_paths=shortest_paths,
    propflow=propflow,
    katz=katz, 
    target=target
  )

  print_status("Store features.") 
  pd.DataFrame(features).to_pickle(args.directory + 'features.pkl')

  