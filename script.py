
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(f, n_dims = 100,sigma_scaler = 1.005,
                                   mu_scaler = 10, elitist_level = 4, ros_flag= False, saving_interval=20)

optimizer.training(N_iterations=100_000, sample_size= 10, save_results= True,filename='q11.npz')

