
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(rastrigin, n_dims = 5000,sigma_scaler = 1.00005,
                                   mu_scaler = 250, elitist_level = 10, ros_flag=False)

optimizer.training(N_iterations=100000, sample_size= 150, sample_increaser_factor=0, save_results= True,filename= 'testing_evl_top_2.npz')

