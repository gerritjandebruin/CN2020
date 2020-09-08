#!/usr/bin/env python

import joblib

from feature_construction import feature_construction

paths = [f'datasets/au/temporal{i}/test/' for i in [1, 2, 3, 4, 5, 6, 8]]

joblib.Parallel(n_jobs=7)(joblib.delayed(feature_construction)(path, n_jobs=40, progress=False, position=index) for index, path in enumerate(paths))