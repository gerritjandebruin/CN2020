#!/usr/bin/env python

import datetime
import math
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

# Typing
NodePair = Tuple[int, int]
Edge = List[Tuple[int, int, Dict['date', datetime.datetime]]]

def read_edges(file: str, sep=' ', skiprows=1) -> pd.DataFrame:
  d = pd.read_csv(file, sep, skiprows=skiprows, names=['source', 'target', 'weight', 'date'])
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


def get_graph(edgelist: pd.DataFrame) -> nx.Graph:
  """Add edge to graph. Contains edge attribute weight."""
  g = nx.Graph()
  
  for u, v, _ in edgelist.itertuples(index=False, name=None):
    weight = g[u][v]["weight"]+1 if g.has_edge(u,v) else 1
    g.add_edge(u, v, weight=weight)
  
  return g


def giant_component(graph: nx.Graph) -> nx.Graph: return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


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