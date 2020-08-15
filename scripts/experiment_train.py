#!/usr/bin/env python

import argparse
import collections
import copy
import datetime
import itertools
import json
import math
import pickle
import time
from typing import List, Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
import seaborn as sns
from tqdm import tqdm
from xgboost import XGBClassifier

# Typing
NodePair = Tuple[int, int]
Edge = List[Tuple[int, int, Dict['date', datetime.datetime]]]

# Constants
basepath = '/local/bruingjde/complexnetworks2020-experiment/'
chunk_size = 1000

# Functions
def flatten(l): return [item for sublist in l for item in sublist]
def print_status(desc: str): print(f'{datetime.strftime("%H:%M:%S", datetime.time.localtime())}: {desc}')
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
## Basic IO
def read_edges(file: str) -> pd.DataFrame:
  d = pd.read_csv(file, sep=' ', skiprows=1, names=['source', 'target', 'weight', 'date'])
  d['date'] = d['date'].apply(datetime.datetime.fromtimestamp)
  d.sort_values(by='date', inplace=True)
  return d.loc[:, ['source', 'target', 'date']]
def filter_edgelist(edges: pd.DataFrame, start=0, stop=1, verbose=True) -> pd.DataFrame: 
  """Filter edgelist.  If start/ stop is float, start/stop from the fraction of total edges. If datetime, this is used.""" 
  no_edges = len(edges)
  if start != 0:
    if type(start) is float:
      assert 0 < start < 1
      start = int(start*no_edges)
    if type(start) is int: start = edges.iloc[start]['date']
    start = start + datetime.timedelta(seconds=1)
  else: start = edges['date'].min()
  if verbose: print(f'{start=}')
  
  if stop != 1:
    if type(stop) is float:
      assert 0 < stop < 1
      stop = math.floor(stop*no_edges)-1
    if type(stop) is int: stop = edges.iloc[stop]['date']
  else: stop = edges['date'].max()
  if verbose: print(f'{stop=}')
  
  mask = (edges['date'] >= start) & (edges['date'] <= stop)
  if verbose: 
    no_selected_edges = sum(mask)
    print(f'{no_selected_edges=} ({no_selected_edges/len(edges):.1e})')

  return edges.loc[mask]
def convert_to_set(edges: pd.DataFrame) -> List[NodePair]: return {edge for edge in edges.loc[:, ['source', 'target']].itertuples(index=False, name=None)}
def get_graph(edgelist: pd.DataFrame) -> nx.Graph:
  """Add edge to graph. Contains edge attribute weight."""
  g = nx.Graph()
  
  for u, v, _ in edgelist.itertuples(index=False, name=None):
    weight = g[u][v]["weight"]+1 if g.has_edge(u,v) else 1
    g.add_edge(u, v, weight=weight)
  
  return g
def giant_component(graph: nx.Graph) -> nx.Graph: return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
def report(graph:nx.Graph, probes: Tuple[int, int]) -> dict:
  print('Determine number of candidates.')
  non_edges = set(nx.non_edges(graph))
  result = dict(
    number_probes=len(probes), 
    candidate=sum([np in non_edges for np in probes]),
    already_edge=sum([graph.has_edge(u, v) for u, v in probes]),
    no_candidate=sum([not (graph.has_node(u) and graph.has_node(v)) for u, v in probes])
  )
  print_status(json.dumps(result, indent=4))
  return result
def get_distances(graph: nx.Graph, cutoff: int = None) -> (List[NodePair], List[int]):
  """
  Get all non-edges using BFS. When cutoff provided, consider only node pairs with at most this distance.
  Returns:
  - nodepairs: tuple containing all nodepairs
  - distances: tuple containing all distances
  """
  return zip(
    *[
      ((u, v), distance)
      for u, (nbs_u, _) in tqdm(nx.all_pairs_dijkstra(graph, cutoff, weight=None), total=len(graph), desc="get_distances")
      for v, distance in nbs_u.items() if distance > 1 and (cutoff is None or distance <= cutoff) 
    ]
  )

## Feature construction
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
def feature_construction_hplp(graph: nx.Graph, nodepairs: List[NodePair], targets: List[bool]) -> pd.DataFrame:
  features = dict()

  # Degree
  degree = np.array([[degree for _, degree in graph.degree(nodepair)] for nodepair in tqdm(nodepairs, desc="Degree")])
  degree.sort(axis=1)
  features['d_min'] = degree[:,0]
  features['d_max'] = degree[:,1]

  # Volume
  volume = np.array([[degree for _, degree in graph.degree(nodepair, weight='weight')] for nodepair in tqdm(nodepairs, desc="Volume")])
  volume.sort(axis=1)
  features['v_min'] = volume[:,0]
  features['v_max'] = volume[:,1] 

  # Common Neighbors
  features['cn'] = np.array([len(list(nx.common_neighbors(graph, *nodepair))) for nodepair in tqdm(nodepairs, desc='Common Neighbors')])

  # Propflow
  print_status("Calculate propflow.")
  score = get_propflow(graph)
  features['pf'] = np.fromiter(((score.get(u, 0).get(v, 0) + score.get(v, 0).get(u, 0))/2 for u, v in tqdm(nodepairs, desc='propflow')), dtype=float)

  # Shortest Paths
  sp_dict = {node: {k: len(v) for k, v in nx.predecessor(graph, node, cutoff=5).items()} for node in tqdm(graph, desc='calculating shortest paths')}
  features['sp'] = np.fromiter((sp_dict[u][v] for u, v in tqdm(nodepairs, desc='Retrieve sp')), dtype=int)

  # Multi-core calculations:
  no_chunks = len(nodepairs) // chunk_size
  nodepair_chuncks = np.array_split(nodepairs, no_chunks)
  
  ## Maxflow
  print_status(f"Build residual network for {'preflow_push' if args.preflow else 'edmonds_karp'}.")
  residual = nx.algorithms.flow.utils.build_residual_network(graph, 'weight')
  
  mf = ProgressParallel(n_jobs=128, total=no_chunks, desc='Maxflow (parallel)')(
    joblib.delayed(get_mf)(graph, nodepair_chunck, residual, flow_func=nx.algorithms.flow.edmonds_karp, cutoff=5) for nodepair_chunck in nodepair_chuncks
  )
  features['mf'] = np.array(flatten(mf))

  features['target'] = target

  return pd.DataFrame(features, index=nodepairs)

## Analyze grid search
def get_x_y(df: pd.DataFrame): return df.drop(columns='target').values, df['target'].values
def gridsearch(df: pd.DataFrame, random_state=1, also_random=True, max_depth=[1, 2]) -> pd.DataFrame:
  X, y = get_x_y(df)
  
  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/3, random_state=random_state)
  clf = XGBClassifier(random_state=random_state, tree_method='hist', n_jobs=6)
  gridsearch = GridSearchCV(
    clf, 
    param_grid=dict(max_depth=max_depth, scale_pos_weight=[sum(~y_train)/sum(y_train), 1]), 
    scoring='average_precision', 
    n_jobs=30,
    cv=StratifiedKFold(shuffle=True, random_state=random_state),
    return_train_score=True
  )
  
  if also_random: 
    gridsearch_random = copy.deepcopy(gridsearch)
    np.random.seed(random_state)
    y_random = copy.deepcopy(y_train)
    np.random.shuffle(y_random)
  
  gridsearch.fit(X_train, y_train)
  df_dict = dict(
      mean_train=gridsearch.cv_results_['mean_train_score'],
      std_train=gridsearch.cv_results_['std_train_score'],
      mean_val=gridsearch.cv_results_['mean_test_score'],
      std_val=gridsearch.cv_results_['std_test_score'],
      val_fold0=gridsearch.cv_results_[f'split0_test_score'],
      val_fold1=gridsearch.cv_results_[f'split1_test_score'],
      val_fold2=gridsearch.cv_results_[f'split2_test_score'],
      val_fold3=gridsearch.cv_results_[f'split3_test_score'],
      val_fold4=gridsearch.cv_results_[f'split4_test_score']
  )
  
  if also_random: 
    gridsearch_random.fit(X_trainval, y_random)
    df_dict['mean_train_random']=gridsearch_random.cv_results_['mean_train_score']
    df_dict['std_train_random']=gridsearch_random.cv_results_['std_train_score']
    df_dict['mean_val_random']=gridsearch_random.cv_results_['mean_test_score']
    df_dict['std_val_random']=gridsearch_random.cv_results_['std_test_score']
  df = pd.DataFrame(df_dict, index=pd.Index([(d['max_depth'], d['scale_pos_weight'] > 1) for d in gridsearch.cv_results_['params']], name=('max_depth', 'balanced')))
  df['diff_train_val'] = df['mean_val'] - df['mean_train']
  df['rstd_test'] = df['std_val'] / df['mean_val']
  if also_random: df['val_over_random'] = df['mean_val'] - df['mean_val_random']
  return df.sort_values('mean_val', ascending=False)
def report_performance(df_train: pd.DataFrame, df_test: pd.DataFrame, random_state=1, max_depth=1, tree_method='hist', balanced=True, n_jobs=128):
  X, y = get_x_y(df_train)
  clf = XGBClassifier(max_depth=max_depth, n_jobs=128, tree_method=tree_method, scale_pos_weight=sum(~y)/sum(y) if balanced else 1 , random_state=random_state)
  clf.fit(X, y)
  X_test, y_test = get_x_y(df_test)
  y_pred = clf.predict_proba(X_test)[:,1]
  return average_precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred)

parser = argparse.ArgumentParser()
parser.add_argument('configuration', help='Location of configuration json.', type=str)
parser.add_argument('output', help='Location of output pickle.', type=str)
args = parser.parse_args()

results = dict()
with open(basepath + args.configuration) as file: configuration = json.load(file)

results['configuration'] = configuration

with open(basepath + configuration['input']) as file:
  edges = read_edges(file=file)
results['total_edges'] = len(edges)
print(f'Number of edges: {len(edges):.1e}')

print('train mature')
edges_train_mature = filter_edgelist(edges, stop=configuration['stop_mature'])
print('train probe')
edges_train_probe = filter_edgelist(edges, start=configuration['stop_mature'], stop=configuration['stop_train'])
print('test mature')
edges_test_mature = filter_edgelist(edges, stop=configuration['stop_train'])
print('test probe')
edges_test_probe = filter_edgelist(edges, start=configuration['stop_train'], stop=configuration['stop_test'])

result['graph'] = giant_component(get_graph(edges_train_mature))
result['probes'] = convert_to_set(edges_train_probe)

result['candidates'] = report(graph=g_train_matured, probes=uv_train_probe)

report(graph=result['graph'], probes=result['probes'])

# TRAIN
nodepairs_train, _ = get_distances(graph=result['graph'], cutoff=configuration['distance'])
targets_train = [nodepair in uv_train_probe for nodepair in tqdm(nodepairs_train)]

result['features'] = feature_construction_hplp(graph=g_train_matured, nodepairs=nodepairs_train)

result['gridsearch_train'] = gridsearch(features)
print_status(result['gridsearch_train'])

print_status('Store results')
with args.output.open('w') as file:
  pickle.dump(result, args.output, protocol=pickle.HIGHEST_PROTOCOL)


