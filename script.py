
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(rastrigin, n_dims = 100,sigma_scaler = 1.0001,
                                   mu_scaler = 100, elitist_level = 4, ros_flag=False, saving_interval=20)

optimizer.training(N_iterations=4_000_000, sample_size= 10, save_results= True,filename='q10.npz')

#Error aj