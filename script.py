
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(g, n_dims = 100,sigma_scaler = 1.0008,
                                   mu_scaler = 100, elitist_level = 4, ros_flag=False)

optimizer.training(N_iterations=50000, sample_size= 10, save_results= True,filename='q0.npz')

