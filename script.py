
from qea import QuantumEvAlgorithm
from ttest_functions import f,g,rastrigin,rosenbrock, griewank, michael, schwefel, dropwave, schaffer_2, equation


optimizer = QuantumEvAlgorithm(equation, n_dims = 20,sigma_scaler = 1.0009,
                                   mu_scaler = 20, elitist_level = 15, ros_flag= False, saving_interval=500)

optimizer.training(N_iterations=150000, sample_size= 150, save_results= True,filename='q11.npz')




