import numpy as np
from qea import QuantumEvAlgorithm
import time
import os
from main import f,g,rastrigin,rosenbrock

optimizer = QuantumEvAlgorithm(f, n_dims = 50,sigma_scaler = 1.0005,
                                   mu_scaler = 50, elitist_level = 2)
optimizer.training(N_iterations=200000, sample_size= 10)