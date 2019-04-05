

import numpy as np
def sigmoid(z):
    g = 1/(1.0 + np.exp(-z)) 
    return(g)