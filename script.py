
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 300,sigma_scaler = 1.002,
                                   mu_scaler = 40, elitist_level = 3, ros_flag=False)

optimizer.training(N_iterations=10000, sample_size= 100, save_results= False,filename='qea_testing_5000_50.npz')


# Mañana comenzar a sacar resultados. Me gustaría.