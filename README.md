# Link prediction

## Features
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

## Parameter tuning
- eval_metric doesn't matter
- balanced only necessary for n>2
- n_estimator default is okay
- max_depth=1
