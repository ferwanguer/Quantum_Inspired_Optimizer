
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave


optimizer = QuantumEvAlgorithm(dropwave, n_dims = 2,sigma_scaler = 1.007,
                                   mu_scaler = 15, elitist_level = 5, ros_flag= False, saving_interval=5)

optimizer.training(N_iterations=1000, sample_size= 20, save_results= True,filename='q11.npz')

