#!/usr/bin/env python

import argparse
from time import localtime, strftime

import joblib
import pandas as pd

def print_status(desc: str): 
  print(f'{strftime("%H:%M:%S", localtime())}: {desc}')

if __name__ == "__main__":
  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'directory', 
    help='Location where targets.pkl, singlecore.pkl, propflow.pkl, maxflow.pkl, shortest_paths and katz.pkl are present. Result is stored as features.pkl in this directory.')
  args = parser.parse_args()

  print_status('Load single core features')
  features = joblib.load(args.directory + 'singlecore.pkl')
  
  print_status('Load target')
  features['target'] = joblib.load(args.directory + 'targets.pkl')

  print_status('Load maxflow features')
  features['mf'] = joblib.load(args.directory + 'maxflow.pkl')

  print_status("Store features.") 
  pd.DataFrame(features).to_pickle(args.directory + 'features.pkl')