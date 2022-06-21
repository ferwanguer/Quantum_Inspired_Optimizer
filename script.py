
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock

optimizer = QuantumEvAlgorithm(f, n_dims = 5000,sigma_scaler = 1.00005,
                                   mu_scaler = 100, elitist_level = 2)

optimizer.training(N_iterations=300000, sample_size= 10, sample_increaser_factor=0,save_results= True ,filename= 'testing_evl.npz')