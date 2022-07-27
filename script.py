
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 50,sigma_scaler = 1.001,
                                   mu_scaler = 80, elitist_level = 2, ros_flag=False)

optimizer.training(N_iterations=100000, sample_size= 3, save_results= True,filename='qea_testing.npz')

# Mañana comenzar a sacar resultados. Me gustaría.