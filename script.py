
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(rastrigin, n_dims = 10000,sigma_scaler = 1.000002,
                                   mu_scaler = 5, elitist_level = 10, ros_flag= False, saving_interval=20)

optimizer.training(N_iterations=10_000_0, sample_size= 100, save_results= True,filename='q11.npz')

