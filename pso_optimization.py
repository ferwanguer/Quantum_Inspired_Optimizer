from test_functions import f, g, rastrigin, rosenbrock, michael, schwefel, dropwave, schaffer_2
from pyswarms.single import GlobalBestPSO
import numpy as np
import time
import os
#Testing the activity commit
start = time.time()
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n_dimensions = 2
x_max = 10 * np.ones(n_dimensions)
x_min = -10* np.ones(n_dimensions)
bounds = (x_min, x_max)
iterations = 5000
n_particles = 8000
optimizer = GlobalBestPSO(n_particles= n_particles, dimensions = n_dimensions,options = options, bounds= bounds)
optimizer.optimize(schaffer_2,iters = iterations)

cost_history = optimizer.cost_history
position_history = optimizer.pos_history[0]

end = time.time()
function_evaluations = iterations*n_particles


optimization_time = np.linspace(0, function_evaluations, num= len(cost_history))
print(f'The PSO algorithm took {end-start} seconds')


results_path = 'Results'
np.savez(os.path.join(results_path,"testing_pso_500.npz"), cost_history,position_history,optimization_time ,cost_h = cost_history,
         time = optimization_time, pos_history = position_history)
