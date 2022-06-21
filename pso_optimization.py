from main import rosenbrock
from pyswarms.single import GlobalBestPSO
import numpy as np
import time
import os
start = time.time()
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n_dimensions = 50
x_max = 5.12 * np.ones(n_dimensions)
x_min = - x_max
bounds = (x_min, x_max)
iterations = 1000
n_particles = 3000

optimizer = GlobalBestPSO(n_particles= n_particles, dimensions = n_dimensions,options = options, bounds= bounds)
optimizer.optimize(rosenbrock,iters = iterations)

cost_history = optimizer.cost_history
position_history = optimizer.pos_history[0]
print(position_history)
end = time.time()
function_evaluations = iterations*n_particles

# optimization_time = np.linspace(0, end - start, num= len(cost_history))

optimization_time = np.linspace(0, function_evaluations, num= len(cost_history))
print(f'The PSO algorithm took {end-start} seconds')
results_path = 'Results'
np.savez(os.path.join(results_path,"Optimization_results_rastrigin_pso_ev_50dim_1.npz"), cost_history,position_history,optimization_time ,cost_h = cost_history,
         time = optimization_time, pos_history = position_history)
print('End')