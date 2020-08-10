#!/usr/bin/env python

import datetime
import os

import joblib
import networkx as nx
import pandas as pd

def giant_component(graph: nx.Graph) -> nx.Graph: return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

folder = '/local/bruingjde/complexnetworks2020-experiment/src'

d = pd.concat([pd.read_csv(f'{folder}/artexhibit/{file}', sep='\t', names=['date', 'source', 'target'], dtype=int) for file in os.listdir(f'{folder}/artexhibit/')], ignore_index=True)
d['date'] = d['date'].apply(datetime.datetime.fromtimestamp)
d = d.sort_values('date')
d = d[['source', 'target', 'date']]
g = nx.from_pandas_edgelist(d)
print(f'number of edges (GC): {g.number_of_edges()} ({giant_component(g).number_of_edges()})')

d = [(u, v, dict(date=date)) for u, v, date in d.itertuples(index=False)]
joblib.dump(d, f'{folder}/artexhibit.pkl')