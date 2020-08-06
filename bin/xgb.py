#!/usr/bin/env python

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils import shuffle
from tqdm import tqdm
from xgboost import XGBClassifier


random_state = 1
split_params = dict(test_size=1/3, random_state=random_state)
clf_params = dict(random_state=random_state, tree_method='hist', n_jobs=6)
grid_params = dict(
  param_grid=dict(max_depth=[1, 2], class_weight=['balanced', None]), 
  scoring='average_precision', 
  n_jobs=30,
  cv=StratifiedKFold(shuffle=True, random_state=random_state), 
  verbose=10, 
  return_train_score=True
)
filename = f"xgb"

def print_status(desc: str): print(f'{time.strftime("%H:%M:%S", time.localtime())}: {desc}')
def _convert_grid_params(grid_params: dict, y: np.array) -> dict: 
  """Replace class_weight by scale_pos_weight for xgboost.""" 
  if 'class_weight' not in grid_params['param_grid']: return grid_params
  assert set(grid_params['param_grid']['class_weight']) <= {"balanced", None}
  if 'balanced' in grid_params['param_grid'].get('class_weight'):
    grid_params['param_grid']['scale_pos_weight'] = [sum(~y)/sum(y)]
    if None in grid_params['param_grid']['class_weight']:
      grid_params['param_grid']['scale_pos_weight'].append(1)
  del grid_params['param_grid']['class_weight']
  return grid_params
   
if __name__ == "__main__":
  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('filepath', help='Directory where features.pkl is present. Output will be written to the same directory.')
  parser.add_argument('--random', action='store_true', default=False, help='If flag is added, randomly permute y.')
  args = parser.parse_args()
  print_status(f'{args=}')
  print_status(f'Result will be stored at {args.filepath}{filename}.pkl')

  print_status(f"Read features from {args.filepath}features.pkl")
  df = pd.read_pickle(f'{args.filepath}features.pkl')
  
  print_status("Split into X, y")
  X = df.drop(columns='target').values
  y = df['target'].values
  
  np.random.seed(0)
  if args.random: np.random.shuffle(y)  
  
  grid_params = _convert_grid_params(grid_params, y)
  
  # Trainval and test split
  X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, **split_params)
  
  # Run
  clf = XGBClassifier(**clf_params)
  result = GridSearchCV(clf, **grid_params)
  result.fit(X_trainval, y_trainval)
  result.split_params = split_params
  result.clf_params = clf_params
  result.grid_params = grid_params
  # Store results
  print_status("Store results")
  joblib.dump(result, f"{args.filepath}{filename}.pkl")