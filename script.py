
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 200,sigma_scaler = 1.0001,
                                   mu_scaler = 80, elitist_level = 3, ros_flag=False)

optimizer.training(N_iterations=200000, sample_size= 4, save_results= False,filename='nPondered.npz')

# Mañana comenzar a sacar resultados. Me gustaría.