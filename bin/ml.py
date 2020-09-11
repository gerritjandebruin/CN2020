import copy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from xgboost import XGBClassifier

def gridsearch(df: pd.DataFrame, random_state=1, also_random=True, max_depth=[1, 2]) -> pd.DataFrame:
  X = df.drop(columns='target').values
  y = df['target'].values
  
  param_grid=dict(max_depth=max_depth, scale_pos_weight=[sum(~y)/sum(y), 1])
  
  X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=1/3, random_state=random_state)
  clf = XGBClassifier(random_state=random_state, tree_method='hist', n_jobs=6)
  gridsearch = GridSearchCV(
    clf, 
    param_grid=param_grid, 
    scoring='average_precision', 
    n_jobs=30,
    cv=StratifiedKFold(shuffle=True, random_state=random_state),
    return_train_score=True
  )
  
  if also_random: 
    gridsearch_random = copy.deepcopy(gridsearch)
    np.random.seed(random_state)
    y_random = copy.deepcopy(y_trainval)
    np.random.shuffle(y_random)
  
  gridsearch.fit(X_trainval, y_trainval)
  df_dict = dict(
      mean_train=gridsearch.cv_results_['mean_train_score'],
      std_train=gridsearch.cv_results_['std_train_score'],
      mean_test=gridsearch.cv_results_['mean_test_score'],
      std_test=gridsearch.cv_results_['std_test_score'],
      test_fold0=gridsearch.cv_results_[f'split0_test_score'],
      test_fold1=gridsearch.cv_results_[f'split1_test_score'],
      test_fold2=gridsearch.cv_results_[f'split2_test_score'],
      test_fold3=gridsearch.cv_results_[f'split3_test_score'],
      test_fold4=gridsearch.cv_results_[f'split4_test_score']
  )
  
  if also_random: 
    gridsearch_random.fit(X_trainval, y_random)
    df_dict['mean_train_random']=gridsearch_random.cv_results_['mean_train_score']
    df_dict['std_train_random']=gridsearch_random.cv_results_['std_train_score']
    df_dict['mean_test_random']=gridsearch_random.cv_results_['mean_test_score']
    df_dict['std_test_random']=gridsearch_random.cv_results_['std_test_score']
  df = pd.DataFrame(df_dict, index=pd.Index([(d['max_depth'], d['scale_pos_weight'] > 1) for d in gridsearch.cv_results_['params']], name=('max_depth', 'balanced')))
  df['diff_train_test'] = (df['mean_test'] - df['mean_train']).abs()
  df['rstd_test'] = df['std_test'] / df['mean_test']
  if also_random: df['test_over_random'] = df['mean_test'] - df['mean_test_random']
  return df.sort_values('mean_test', ascending=False)
    
def get_x_y(df: pd.DataFrame): return df.drop(columns='target').values, df['target'].values
def report_performance(df_train: pd.DataFrame, df_test=None, random_state=1, max_depth=1, tree_method='hist', balanced=True, n_jobs=256):
  X, y = get_x_y(df_train)
  if df_test is None: X, X_test, y, y_test = train_test_split(X, y, test_size=1/3, random_state=random_state)
  else: X_test, y_test = get_x_y(df_test)
  clf = XGBClassifier(max_depth=max_depth, n_jobs=128, tree_method=tree_method, scale_pos_weight=sum(~y)/sum(y) if balanced else 1 , random_state=random_state)
  clf.fit(X, y)
  y_pred = clf.predict_proba(X_test)[:,1]
  return dict(ap=average_precision_score(y_test, y_pred), roc=roc_auc_score(y_test, y_pred))