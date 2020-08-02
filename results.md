# Random split
| Experiment | Maturing phase | Probe phase  | Total nodepairs | Positive    | Probe set*  | Test score | Feature construction |
|:----------:|:--------------:|:------------:|:---------------:|:-----------:|:-----------:|:----------:|:--------------------:|
| A1         | <= 97 (~33%)   | 98 => (~67%) | 27M             | 4199 (2e-4) |  8% 10% 82% | 0.0049(9)  | 25h @ viridium       |
| A2         | <= 97 (~33%)   | 98-99 (~42%) | 27M             | 2908 (1e-4) | 12% 10% 78% | 0.0045(6)  | ~25h @ viridium      |
| A3         | <= 99 (~75%)   | 00 => (~25%) | 138M            | 4035 (3e-5) | 25% 20% 55% | **         |

# Random split
| Experiment | Maturing phase | Probe phase  | Total nodepairs | Positive    | Probe set*  | Test score | Feature construction |
|:----------:|:--------------:|:------------:|:---------------:|:-----------:|:-----------:|:----------:|:--------------------:|
| A1         | <= 97 (~33%)   | 98 => (~67%) | 27M             | 4199 (2e-4) |  8% 10% 82% | 0.0049(9)  | 25h @ viridium       |
| A2         | <= 97 (~33%)   | 98-99 (~42%) | 27M             | 2908 (1e-4) | 12% 10% 78% | 0.0045(6)  | ~25h @ viridium      |
| A3         | <= 99 (~75%)   | 00 => (~25%) | 138M            | 4035 (3e-5) | 25% 20% 55% | **         |

# Temporal split
| Experiment | Maturing phase (train) | Probe phase (train) | Maturing phase (test) | Probe phase (test) | Test score |
|:----------:|:----------------------:|:-------------------:|:---------------------:|:------------------:|:----------:|
| B1         | <= 97 (~33%)           | 98-99 (~42%)        | <= 99 (~75%)          | 00 => 25%          |            |

* already in maturing phase, candidate, one of nodes not in graph
** computational too expensive

# Hyperparameter optimalization
- max_depth = 1 is better than higher values (equal val score, lower train score)
- tree_method 'hist' give similiar results as 'exact' and 'approx' but is faster
- scaling does not improve accuracy