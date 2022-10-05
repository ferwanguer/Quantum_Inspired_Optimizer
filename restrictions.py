import numpy as np 
def h(x: np.ndarray):

    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    return  -x[:,1 ] - x[:,0] + 6

def h_1(x: np.ndarray):

    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    return  -x[:,2] - x[:,0] + 6