
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2


optimizer = QuantumEvAlgorithm(schaffer_2, n_dims = 2,sigma_scaler = 1.0001,
                                   mu_scaler = 1, elitist_level = 1, ros_flag= False, saving_interval=5)

optimizer.training(N_iterations=100_000, sample_size= 150, save_results= True,filename='q11.npz')





#Ojala...