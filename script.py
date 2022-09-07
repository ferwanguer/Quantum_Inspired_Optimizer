
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel


optimizer = QuantumEvAlgorithm(g, n_dims = 4500,sigma_scaler = 1.0001,
                                   mu_scaler = 5, elitist_level = 3, ros_flag= False, saving_interval=100)

optimizer.training(N_iterations=100000, sample_size= 10, save_results= True,filename='q11.npz')

