from datetime import datetime
import itertools
from typing import List, Any, Dict, Tuple

import networkx as nx
from tqdm import tqdm

# Typing
Author = int
Authors = List[Author]
Year = int
Paper = Tuple[Authors, Year]
Edge = Tuple[Author, Author, Dict[str, Year]]
Edges = List[Edge]
NodePair = Tuple[int, int]
NodePairs = List[NodePair]
Distances = List[int]

def get_papers(start: int = None, stop: int = None, filepath: str = "src/cond-mat.hg2") -> List[Paper]:
  """Read collaboration data in filepath and return all papers."""
  
  papers = list()
  # Get number of rows to read for the vertices.
  with open(filepath) as file:
    no_rows = int(file.readline().split(' ')[1])
 
  with open(filepath) as file:
    for paper in file.readlines()[no_rows+2:]:
      # Each line has the following format: epoch no_authors [ u v (w ...) ]
      epoch = datetime.fromtimestamp(int(paper.split(' ')[0]))
      year = epoch.year
      if start is not None and year < start: continue
      elif stop is not None and year > stop: return papers
          
      no_authors = int(paper.split(' ')[1])
      index1 = paper.find('[')+2
      index2 = paper.find(']')-1

      authors = [int(auth) for auth in paper[index1:index2].split(' ')]
      assert no_authors == len(authors)
      
      papers.append((authors, epoch))
  return papers


def get_edgelist(papers: List[Paper] = None, **kwargs) -> Edges:
  """
  Return edgelist from data. If data is not provided this is automatically 
  created using get_data. **kwargs are provided to this function.
  """
  if papers is None: papers = get_papers(**kwargs)
    
  return [
    (u, v, dict(date=date)) if u<v else (v, u, dict(date=date))
    for authors, date in papers
    for u, v in itertools.combinations(authors, 2)
  ]


def giant_component(graph: nx.Graph) -> nx.Graph: return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def get_graph(edgelist: Edges) -> nx.Graph:
  """Add edge to graph. Contains edge attribute weight."""
  g = nx.Graph()
  
  for u, v, _ in edgelist:
    weight = g[u][v]["weight"]+1 if g.has_edge(u,v) else 1
    g.add_edge(u, v, weight=weight)
  
  return g


def get_distances(graph: nx.Graph, cutoff: int = None) -> (NodePairs, Distances):
  """
  Get all non-edges using BFS. When cutoff provided, consider only node pairs with at most this distance.
  Returns:
  - nodepairs: tuple containing all nodepairs
  - distances: tuple containing all distances
  """
  return zip(
    *[
      [(u, v), distance]
      for u, (nbs_u, _) in tqdm(nx.all_pairs_dijkstra(graph, cutoff, weight=None), total=len(graph), desc="get_distances")
      for v, distance in nbs_u.items() if distance > 1 and (cutoff is None or distance < cutoff) 
    ]
  )