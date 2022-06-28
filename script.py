
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock

optimizer = QuantumEvAlgorithm(f, n_dims = 2,sigma_scaler = 1.01,
                                   mu_scaler = 10, elitist_level = 1, ros_flag=False)

optimizer.training(N_iterations=100000, sample_size= 2, sample_increaser_factor=0,save_results= True ,filename= 'testing_evl.npz')