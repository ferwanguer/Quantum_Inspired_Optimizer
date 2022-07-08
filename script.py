
from qea import QuantumEvAlgorithm
from test_functions import f,g,rastrigin,rosenbrock


optimizer = QuantumEvAlgorithm(g, n_dims = 50,sigma_scaler = 1.001,
                                   mu_scaler = 80, elitist_level = 5, ros_flag=False)

optimizer.training(N_iterations=70000, sample_size= 10, sample_increaser_factor=0, save_results= True,filename= 'testing_evl_top_2bis.npz')

