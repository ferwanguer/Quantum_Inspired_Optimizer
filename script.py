import numpy as np
from qea import QuantumEvAlgorithm
import time
import os
from test_functions import f,g,rastrigin,rosenbrock

optimizer = QuantumEvAlgorithm(f, n_dims = 500,sigma_scaler = 1.0005,
                                   mu_scaler = 50, elitist_level = 2)

optimizer.training(N_iterations=150000, sample_size= 10, sample_increaser_factor=0,save_results= False ,filename= 'testing_evl.npz')