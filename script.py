import numpy as np 
import pandas
from qea import QuantumEvAlgorithm
from ttest_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2, equation

up = 5*np.ones(7)
low = -5*np.ones(7)
integrals = np.array([True,True,False, False,False,False,True])
optimizer = QuantumEvAlgorithm(f, n_dims = 7,upper_bound= up, lower_bound= low, integral_id=integrals, sigma_scaler = 1.03,
                                   mu_scaler = 20, elitist_level = 1)

results = optimizer.training(N_iterations=1000, sample_size= 20, save= False ,filename='q11.npz')

print(results["min"])


#Todo en orden