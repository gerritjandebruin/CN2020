#!/usr/bin/env python

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils import shuffle
from tqdm import tqdm
from xgboost import XGBClassifier


random_state = 1

n_jobs_outer_cv = 1
n_jobs_inner_cv = 5
n_jobs_xgboost = 25

n_splits_outer = 5
n_splits_inner = 5
max_depths = [1]
tree_methods = ['hist']
scalers = [
  StandardScaler(), 
  MinMaxScaler(), 
  RobustScaler(), 
  PowerTransformer(method='yeo-johnson'), 
  PowerTransformer(method='box-cox'), 
  QuantileTransformer(output_distribution='uniform', random_state=random_state),
  QuantileTransformer(output_distribution='normal', random_state=random_state)
]
filename = f'xgb_hist_maxdepth12_scalers'

def print_status(desc: str): print(f'{time.strftime("%H:%M:%S", time.localtime())}: {desc}')
  
  
def cross_validation(X, y, params, verbose=True):
  estimator = XGBClassifier(n_jobs=n_jobs, **params)
  cv = StratifiedKFold(shuffle=True, random_state=random_state)
  return cross_validate(estimator, X, y, scoring='average_precision', cv=cv, verbose=10, fit_params=dict(verbose=True), return_train_score=True, return_estimator=True)


def nested_cross_validation(X, y, param_grid, params):
  outer_cv = StratifiedKFold(n_splits_outer, shuffle=True, random_state=random_state).split(X, y)
  inner_cv = StratifiedKFold(n_splits_inner, shuffle=True, random_state=random_state)
  
  pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(**params))
  ])
  
  clf_fold = GridSearchCV(
    pipe, 
    param_grid=param_grid,
    scoring='average_precision',
    n_jobs=n_jobs_inner_cv,
    cv=inner_cv,
    refit=True,
    verbose=0,
    return_train_score=True
   )
  
  scores = cross_validate(
    clf_fold, X, y, 
    scoring='average_precision', 
    cv=outer_cv, 
    n_jobs=n_jobs_outer_cv, 
    verbose=10, 
    return_train_score=True, 
    return_estimator=True
  )
  return scores


def main(filepath, nested_cv, random):
  # Load data
  print_status(f"Read features from {filepath + 'features.pkl'}")
  df = pd.read_pickle(filepath + 'features.pkl')
  
  print_status("Split into X, y")
  X = df.drop(columns='target').values
  y = df['target'].values
  
  np.random.seed(0)
  if random: np.random.shuffle(y)  
  
  scale_pos_weight=sum(~y)/sum(y)
  print_status(f'scale_pos_weight: {scale_pos_weight}')
  
  param_grid = dict(classifier__max_depth=max_depths, classifier__tree_method=tree_methods, scaler=scalers)
  params = dict(scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=n_jobs_xgboost)
  
  # Run
  results = nested_cross_validation(X, y, param_grid, params) if nested_cv else cross_validation(X, y, params) 
  
  # Store results
  print_status("Store results")
  joblib.dump(results, f"{filepath}{filename}.pkl")
  
  
if __name__ == "__main__":
  # Get parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('filepath', help='Directory where features.pkl is present. Output will be written to the same directory.')
  parser.add_argument('--nested_cv', action='store_true', default=False, help='If flag is added, apply nested cross-validation. ')
  parser.add_argument('--random', action='store_true', default=False, help='If flag is added, randomly permute y.')
  args = parser.parse_args()
  print_status(f'Nested: {args.nested_cv}, Random: {args.random}')
  print_status(f'Result will be stored at {args.filepath}{filename}.pkl')
  print_status(f'{max_depths=}')
  print_status(f'{tree_methods=}')
  print_status(f'{scalers=}')
  main(filepath=args.filepath, nested_cv=args.nested_cv, random=args.random)