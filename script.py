
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(rastrigin, n_dims = 50,sigma_scaler = 1.00005,
                                   mu_scaler = 60, elitist_level = 3, ros_flag=False)

optimizer.training(N_iterations=500000, sample_size= 10, sample_increaser_factor=0,save_results= True ,filename= 'testing_evl.npz')

