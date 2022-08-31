
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel


optimizer = QuantumEvAlgorithm(michael, n_dims = 2,sigma_scaler = 1.005,
                                   mu_scaler = 5, elitist_level = 2, ros_flag= False, saving_interval=20)

optimizer.training(N_iterations=1000, sample_size= 5, save_results= True,filename='q11.npz')

