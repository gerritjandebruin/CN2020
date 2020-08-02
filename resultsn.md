# Random split
| Experiment | Maturing phase | Probe phase | Total nodepairs | Positive      | Probe set*  | Feature construction | Test score |
|:----------:|:--------------:|:-----------:|:---------------:|:-------------:|:-----------:|:--------------------:|:----------:|
| A1         | 0-33%          | 33-100%     | 41M             | 4668 (1.1e-4) | 11% 12% 77% | ~24h                 |            |
| A2         | 0-33%          | 33-67%      | 41M             | 2760 (6.7e-5) | 16% 13% 71% | ~24h                 |            |
| A3         | 0-67%          | 67-100%     |                 |               | 23%         |                      | **         |

# Temporal split
| Experiment | Maturing phase | Probe phase | Total nodepairs | Positive      | Probe set*  | Feature construction | Test score |
|:----------:|:--------------:|:-----------:|:---------------:|:-------------:|:-----------:|:--------------------:|:----------:|
| B1-train   | 0-33%          | 33-67%      | (A2)
| B1-test    | 0-67%          | 67-100%     | (A3)

* already in maturing phase, candidate, one of nodes not in graph
** computational too expensive