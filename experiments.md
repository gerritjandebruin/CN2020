At best validation score while train score Q1 > median validation score.

# Lightgbm
| Experiment | Maturing | Assessing                            | Learning rate | Num leaves | Iterations | Distance | Train score       | Validation score  | Test score |
|:----------:|:--------:|:------------------------------------:|:-------------:|:----------:|:----------:|:--------:|:-----------------:|:-----------------:|:----------:|
| A1         | <=1997   | >= 1998 Train + test nested CV       | 0.1           | 2          | 26         | 2        | 0.021/0.023/0.025 | 0.013/0.022/0.031 | 0.016      |
| A2         | <=1997   | >=1998 <=1999 Train + test nested CV | 0.1           | 2          | 17         | 2        | 0.017/0.018/0.019 | 0.011/0.016/0.022 | 0.013      |
| B1         | <=1997   | >=1998 <=1999 train | >=2000 test    | 0.1           | 2          | 6          | 2        | 0.012/0.012/0.013 | 0.011/0.012/0.013 | 0.007      |

At same params model.

| Experiment | Maturing | Assessing                            | Learning rate | Num leaves | Iterations | Distance | Train score       | Validation score  | Test score |
|:----------:|:--------:|:------------------------------------:|:-------------:|:----------:|:----------:|:--------:|:-----------------:|:-----------------:|:----------:|
| A1         | <=1997   | >= 1998 Train + test nested CV       | 0.1           | 2          | 6          | 2        | 0.017/0.018/0.019 | 0.011/0.017/0.021 | 0.013      |
| A2         | <=1997   | >=1998 <=1999 Train + test nested CV | 0.1           | 2          | 6          | 2        | 0.014/0.015/0.016 | 0.010/0.014/0.018 | 0.012      |
| B1         | <=1997   | >=1998 <=1999 train | >=2000 test    | 0.1           | 2          | 6          | 2        | 0.012/0.012/0.013 | 0.011/0.012/0.013 | 0.007      |

Random behaviour
| Experiment | Maturing | Assessing                            | Learning rate | Num leaves | Iterations | Distance | Train score       | Validation score  |
|:----------:|:--------:|:------------------------------------:|:-------------:|:----------:|:----------:|:--------:|:-----------------:|:-----------------:|
| A1         | <=1997   | >= 1998 Train + test nested CV       | 0.1           | 2          | 0          | 2        | 0.009/0.010/0.011 |                   | 
| A2         | <=1997   | >=1998 <=1999 Train + test nested CV | 0.1           | 2          | 6          | 2        |                   |                   |
| B1         | <=1997   | >=1998 <=1999 train | >=2000 test    | 0.1           | 2          | 6          | 2        |                   |                   |

# XGBoost
| Experiment | Maturing     | Probe         | Val            | Train          | Time | Balance | Remarks                    |
|:----------:|:------------:|:-------------:|:--------------:|:--------------:|:----:|:-------:|:--------------------------:|
| A1         | <=1997 (33%) | >= 1998 (67%) | .004/.005/.006 | .020/.021/.022 | 18m  | Yes     | Overfitting!               |  