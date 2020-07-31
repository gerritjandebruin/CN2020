#!/usr/bin/env python

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils import shuffle
from tqdm import tqdm
from xgboost import XGBClassifier

random_state = 1

n_jobs_clf = 200
verbose = 1

outer_cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
inner_cv = StratifiedKFold(5, shuffle=True, random_state=random_state)

grid_params = {
  'classifier__class_weight': [None, 'balanced'],
  'scaler': [
    None,
    StandardScaler(), 
    MinMaxScaler(), 
    RobustScaler(), 
    PowerTransformer(method='yeo-johnson'), 
    QuantileTransformer(output_distribution='uniform', random_state=random_state),
    QuantileTransformer(output_distribution='normal', random_state=random_state)
  ]
}

pipe = Pipeline([
  ('scaler', None),
  ('classifier', LogisticRegressionCV(
      cv=inner_cv, 
      scoring='average_precision', 
      penalty='elasticnet',
      solver='saga', 
      random_state=random_state,
      verbose=1,
      max_iter=10000,
      n_jobs=5,
      refit=True,
      l1_ratios=[.1, .5, .7, .9, .95, .99, 1]
    )
  )
])

filename = f'lrcv'

def print_status(desc: str): print(f'{time.strftime("%H:%M:%S", time.localtime())}: {desc}')

def train_clf(X, y, params):
  for train_index, test_index in outer_cv.split(X, y):
    train_index, test_index = next(outer_cv.split(X, y))
    pipe.set_params(**params)
    result = pipe.fit(X[train_index], y[train_index])
    joblib.dump(result)
    break
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('filepath', help='Directory where features.pkl is present. Output will be written to the same directory.')
  args = parser.parse_args()
  filepath = args.filepath
  
  print_status(f"Read features from {filepath}X,y.pkl")
  X = joblib.load(f"{filepath}X.pkl")
  y = joblib.load(f"{filepath}y.pkl") 
  joblib.Parallel(n_jobs=14, verbose=1)(joblib.delayed(train_clf)(X, y, params) for params in list(ParameterGrid(grid_params)))