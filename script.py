
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(griewank, n_dims = 200,sigma_scaler = 1.0002,
                                   mu_scaler = 50, elitist_level = 4, ros_flag=False)

optimizer.training(N_iterations=100000, sample_size= 10, save_results= True,filename='q0.npz')

