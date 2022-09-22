
from qea import QuantumEvAlgorithm
from ttest_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2, equation


optimizer = QuantumEvAlgorithm(g, n_dims = 5,sigma_scaler = 1.04,
                                   mu_scaler = 20, elitist_level = 4, ros_flag= False, saving_interval=500)

optimizer.training(N_iterations=200, sample_size= 10, save_results= True,filename='q11.npz')




#Todo en orden