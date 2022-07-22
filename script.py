
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 50,sigma_scaler = 1.001,
                                   mu_scaler = 80, elitist_level = 2, ros_flag=False)

optimizer.training(N_iterations=50000, sample_size= 3, save_results= False,filename='qea_testing_sigma_0005.npz')

# Mañana comenzar a sacar resultados. Me gustaría.