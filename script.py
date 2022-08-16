
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock, griewank


optimizer = QuantumEvAlgorithm(griewank, n_dims = 100,sigma_scaler = 1.001,
                                   mu_scaler = 50, elitist_level = 4, ros_flag=False)

optimizer.training(N_iterations=40000, sample_size= 20, save_results= True,filename='q0.npz')

#Vuelta de vacaciones.