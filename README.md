# pyCE

A python implementation of the cross-entropy solver for continuous and/or discrete variables. This code is derivative work of the CEoptim R package [1].

# Dependencies
This code requires numpy and scipy. 

# Usage
```python
import numpy as np
from pyCE import cemethod

# define an arbitrary objective function over continous and/or discrete parameters
def myfun(x,cats):
    a = 0
    if 1 == cats[0]:
        a += 2.0
    if 'b' == cats[1]:
        a += 2.0

    return np.square(x).sum()+a

# define the initial sampling distribution for the continuous variables
# this also defines the number of continuous 
cont = {
  'mu':np.ones(1)*4, # mean
  'sigma':np.ones(1)*2, # standard deviation
  'smooth_mean' : 1.0, # smoohing parameter for mean
  'smooth_sd' : 1.0, # smoohing parameter for the standard deviation
  'sd_threshold' : 0.1 # threshold for largest standard deviation
}

# define the space of discrete variables
disc = {
  'categories':[[1,5,2],['a','b','c']] # two variables of arity 3 each
}

# run the solver
_ = cemethod(myfun,maximize=False,continuous = cont,
             discrete = disc,rho=0.1)
```

[1] Benham T., Duan Q., Kroese D.P., Liquet B. (2017) CEoptim: Cross-Entropy R package for optimization. Journal of Statistical Software, 76(8), 1-29.
