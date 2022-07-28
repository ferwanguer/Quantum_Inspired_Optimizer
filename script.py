
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 50,sigma_scaler = 1.0015,
                                   mu_scaler = 80, elitist_level = 4, ros_flag=False)

optimizer.training(N_iterations=40000, sample_size= 8, save_results= True,filename='qea_testingmm.npz')

# Mañana comenzar a sacar resultados. Me gustaría.