#!/usr/bin/env python

import datetime
from itertools import combinations
import math
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

# Typing
NodePair = Tuple[int, int]
Edge = List[Tuple[int, int, Dict['date', datetime.datetime]]]

def construct_edges(file: str): 
  def get_papers(file: str):
    papers = list()
    # Get number of rows to read for the vertices.
    with open(file) as f: no_rows = int(f.readline().split(' ')[1])

    with open(file) as f:
      for paper in f.readlines()[no_rows+2:]:
        # Each line has the following format: epoch no_authors [ u v (w ...) ]
        epoch = datetime.datetime.fromtimestamp(int(paper.split(' ')[0]))

        no_authors = int(paper.split(' ')[1])
        index1 = paper.find('[')+2
        index2 = paper.find(']')-1

        authors = [int(auth) for auth in paper[index1:index2].split(' ')]
        assert no_authors == len(authors)

        papers.append((authors, epoch))
    return papers 
  return pd.DataFrame([(u, v, date) if u<v else (v, u, date) for authors, date in get_papers(file) for u, v in combinations(authors, 2)], columns=['source', 'target', 'date'])

def read_edges(file: str, sep=' ', skiprows=1) -> pd.DataFrame:
  d = pd.read_csv(file, sep, skiprows=skiprows, names=['source', 'target', 'weight', 'date'])
  d['date'] = d['date'].apply(datetime.datetime.fromtimestamp)
  d.sort_values(by='date', inplace=True)
  return d.loc[:, ['source', 'target', 'date']]


def filter_edges(edges: pd.DataFrame, start=0, stop=1, verbose=False) -> pd.DataFrame: 
  """Filter edgelist.  If start/ stop is float, start/stop from the fraction of total edges. If datetime, this is used.""" 
  no_edges = len(edges)
  if start != 0:
    if type(start) is float:
      assert 0 < start < 1
      start = int(start*no_edges)
    if type(start) is int: start = edges.iloc[start]['date']
    start = start + datetime.timedelta(seconds=1)
  else: start = edges['date'].min()
  if verbose: print(start)
  
  if stop != 1:
    if type(stop) is float:
      assert 0 < stop < 1
      stop = math.floor(stop*no_edges)-1
    if type(stop) is int: stop = edges.iloc[stop]['date']
  else: stop = edges['date'].max()
  if verbose: print(stop)
  
  mask = (edges['date'] >= start) & (edges['date'] <= stop)
  if verbose: 
    no_selected_edges = sum(mask)
    print(f'{no_selected_edges=} ({no_selected_edges/len(edges):.1e})')

  return edges.loc[mask]


def convert_to_set(edges: pd.DataFrame) -> List[NodePair]: return {edge for edge in edges.loc[:, ['source', 'target']].itertuples(index=False, name=None)}


def get_graph(edgelist: pd.DataFrame, directed: bool) -> nx.Graph:
  """Add edge to graph. Contains edge attribute weight."""
  g = nx.DiGraph() if directed else nx.Graph()
  
  for u, v, _ in edgelist.itertuples(index=False, name=None):
    weight = g[u][v]["weight"]+1 if g.has_edge(u,v) else 1
    g.add_edge(u, v, weight=weight)
  
  return g


def giant_component(graph): 
  return graph.subgraph(max(nx.strongly_connected_components(graph), key=len)).copy() if type(graph) is nx.DiGraph else graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def report(graph:nx.Graph, probes: Tuple[int, int]):
  n = len(probes)
  print(f"Number of probes: {n}")
  a = sum([graph.has_edge(u, v) for u, v in probes])
  print(f"- already edge: {a} ({a/n:.0%})")
  non_edges = set(nx.non_edges(graph))
  ne = sum([np in non_edges for np in probes])
  print(f"- both nodes in graph: {ne} ({ne/n:.0%})")
  ng = sum([not (graph.has_node(u) and graph.has_node(v)) for u, v in probes])
  print(f"- not in graph: {ng} ({ng/n:.0%})")
  
  
def get_distances(graph: nx.Graph, cutoff: int = None, **kwargs) -> (List[NodePair], List[int]):
  """
  Get all non-edges using BFS. When cutoff provided, consider only node pairs with at most this distance.
  Returns:
  - nodepairs: tuple containing all nodepairs
  - distances: tuple containing all distances
  """
  return zip(
    *[
      ((u, v), distance)
      for u, (nbs_u, _) in tqdm(nx.all_pairs_dijkstra(graph, cutoff, weight=None), total=len(graph), desc="get_distances", **kwargs)
      for v, distance in nbs_u.items() if distance > 1 and (cutoff is None or distance <= cutoff) 
    ]
  )

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

def store(edges, filepath, *, directed: bool, verbose=False, only_do=None, **kwargs):
  if verbose: print_status(filepath)
  for path in ['random/', 'train/', 'test/']: os.makedirs(filepath + path, exist_ok=True)
  if only_do is None or only_do=='random': 
    for name, obj in random(edges, directed=directed, verbose=verbose, **kwargs).items(): joblib.dump(obj, f'{filepath}random/{name}.pkl', protocol=5)
  if only_do is None or only_do=='train':  
    for name, obj in train(edges,  directed=directed, verbose=verbose, **kwargs).items(): joblib.dump(obj, f'{filepath}train/{name}.pkl',  protocol=5)
  if only_do is None or only_do=='test':   
    for name, obj in test(edges,   directed=directed, verbose=verbose, **kwargs).items(): joblib.dump(obj, f'{filepath}test/{name}.pkl',   protocol=5)