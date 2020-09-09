import subprocess
import os.path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def giant_component(graph: nx.Graph) -> nx.Graph:
    """Return the giant component."""
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()


def describe(g: nx.Graph):
  """Give a summary of the graph under study"""
  gc = giant_component(g)
  print(f'Number of edges: {(edges:=g.number_of_edges())}')
  print(f'\t in GC: {(edgesGC:=gc.number_of_edges())} ({edgesGC/edges:.0%})')
  print(f'Number of nodes: {(nodes:=g.number_of_nodes())}')
  print(f'\t in GC: {(nodesGC:=gc.number_of_nodes())} ({nodesGC/nodes:.0%})')
  print(f'Density (in GC): {nx.density(g):.1e} ({nx.density(gc):.1e})')


def degreeDistribution(g: nx.Graph, weight: str = None, num: int = 10) -> None:
  """
  Plot the degree distribution.

  Arguments:
  num: Number of points to use in logbinning the degree distribution. 
  
  """
  degrees = [degree for _, degree in g.degree(weight=weight)]

  counts, bins = np.histogram(
    a = degrees,
    bins = np.round(np.logspace(start=0, stop=np.log10(max(degrees)), num=num))
    )

  x = bins[:-1] + np.diff(bins)/2 # Convert bins to list of midpoints of bins.
  y = counts

  # Fit p0 and p1 in y = p0*x + p1.
  p0, p1 = np.polyfit(x = np.log10(x), y = np.log10(y), deg = 1)

  # Function that transforms any x to y_hat value.
  y_hat = lambda x: 10**(p0*np.log10(x) + p1)

  # Plot
  plt.plot(x, y, marker='o', ls='')
  plt.plot(x, y_hat(x), label=f'y = exp({p0:.2f}*log(x)+{p1:.2f})')

  plt.xscale('log')
  plt.xlabel('Degree')

  plt.yscale('log')
  plt.ylim(1)
  plt.ylabel('Frequency')
  
  plt.legend(
    title=f'Mean: {np.mean(degrees):.0f}, Median: {np.median(degrees):.0f}'
    )
  plt.tick_params(which='both')
  plt.show()


def weightDistribution(
  g: nx.Graph, weight: str = 'weight', yscale: str = 'linear'
  ) -> None:
  """Plot edge-attribute distribution, by default of weight attribute."""
  weights = list(nx.get_edge_attributes(g, weight).values())
  plt.hist(
    x = weights, 
    bins = range(max(weights)), 
    label = f'mean: {np.mean(weights):.1f}, median: {np.median(weights):.1f}'
    )
  plt.yscale(yscale)
  plt.xlim(1)
  plt.xlabel('Weight')
  plt.ylabel('Frequency')
  plt.legend()
  