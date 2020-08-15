#!/usr/bin/env python
import argparse
import datetime
import os

import joblib
import networkx as nx
import pandas as pd

def giant_component(graph: nx.Graph) -> nx.Graph: return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

folder = '/local/bruingjde/complexnetworks2020-experiment/src'

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()

out_files = [file for file in os.listdir(f'datasets/{folder}') if file.startswith('out')][0]
d = pd.read_csv(f'{args.directory}/out.enron', sep=' ', skiprows=1, names=['source', 'target', 'weight', 'date'])
d['date'] = d['date'].apply(datetime.datetime.fromtimestamp)
d = d.sort_values('date')
d = d[['source', 'target', 'date']]
g = nx.from_pandas_edgelist(d)
print(f'number of edges (GC): {g.number_of_edges()} ({giant_component(g).number_of_edges()})')

d = [(u, v, dict(date=date)) for u, v, date in d.itertuples(index=False)]
joblib.dump(d, f'{folder}/enron.pkl')