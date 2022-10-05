import numpy as np 
from qea import QuantumEvAlgorithm
from ttest_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2, equation

up = 5*np.ones(5)
low = -5*np.ones(5)
optimizer = QuantumEvAlgorithm(g, n_dims = 5,upper_bound= up, lower_bound= low,sigma_scaler = 1.04,
                                   mu_scaler = 20, elitist_level = 4, ros_flag= False, saving_interval=500)

optimizer.training(N_iterations=200, sample_size= 15, save_results= False ,filename='q11.npz')




#Todo en orden