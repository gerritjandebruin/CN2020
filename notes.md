# Matrix
|       | Mature  | Probe   |
|:-----:|:-------:|:-------:|
| train | (ta,tb) | (tc,td) |
| test  | (te,tf) | (tg,tg) |

# Features
| Feature | Parameters  | HPLP | HPLP+ |
| Dmax    |             | y    | y     | 
| Dmin    |             | y    | y     |  
| Vmin    |             | y    | y     |
| Vmax    |             | y    | y     |   
| MF      | l=5         | y    | y     |
| SP      | l=5         | y    | y     |
| PF      | l=5         | y    | y     |
| AA      |             |      | y     |
| JC      |             |      | y     |
| KI      | l=5, b=.005 |      | y     |
| PA      |             |      | y     |

# Timing
## Feature construction A1/all:
- mf: ~3days

# Experiments
- A1: Like Lichtenwalter (RandomSplit)
- B1: Like Lichtenwalter (TemporalSplit)

# Parameter tuning
- eval_metric doesn't matter
- balanced only necessary for n>2
- n_estimator default is okay
- max_depth=1

# Current
- b1/4 feature construction @ viridium
- a1/total running @ mithril (crashed)

# To do
- feedback verwerken
- parallelize more
- more datasets
- random forest
- more parameter tuning
- quantify leaking