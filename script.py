
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock

optimizer = QuantumEvAlgorithm(rastrigin, n_dims = 20,sigma_scaler = 1.0001,
                                   mu_scaler = 50, elitist_level = 2, ros_flag=False)

optimizer.training(N_iterations=200000, sample_size= 5, sample_increaser_factor=0,save_results= True ,filename= 'testing_evl.npz')