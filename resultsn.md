# Random split
| Experiment | Maturing phase | Probe phase | Total nodepairs | Positive      | Probe set*  | Test score | Feature construction |
|:----------:|:--------------:|:-----------:|:---------------:|:-------------:|:-----------:|:----------:|:--------------------:|
| A1         | 33%            | 67%         | 41M             | 4668 (1.1e-4) | 11% 12% 77% |            |                      |
| A2         | 33%            | 33%         | 41M             | 2760 (6.7e-5) | 16% 13% 71% |            |                      |
| A3         | 75%            | 25%         |                 | 3623 (      ) | 27% 20% 53% | **         |

* already in maturing phase, candidate, one of nodes not in graph
** computational too expensive

# Random split
| Experiment | Test score |
|:----------:|:----------:|
| A1         | 0.0049(9)  |
| A2         | 0.0045(6)  |
| A3         | **         |

# Hyperparameter optimalization
- max_depth = 1 is better than higher values (equal val score, lower train score)
- tree_method 'hist' give similiar results as 'exact' and 'approx' but is faster
- scaling does not improve accuracy