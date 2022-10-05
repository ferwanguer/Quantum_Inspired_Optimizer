import numpy as np 
import pandas
from qea import QuantumEvAlgorithm
from restrictions import h, h_1
from ttest_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2, equation
n_dims = 50
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims,False)

# np.array([True,True,False, False,False,False,True])

optimizer = QuantumEvAlgorithm(f, n_dims = n_dims,upper_bound= up, lower_bound= low, integral_id=integrals, sigma_scaler = 1.003,
                                   mu_scaler = 20, elitist_level = 6, restrictions=[])

results = optimizer.training(N_iterations=25500, sample_size= 20, save= False ,filename='q11.npz')

print(results["min"])


#Todo en orden