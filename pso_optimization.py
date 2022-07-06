from test_functions import f, g, rastrigin, rosenbrock
from pyswarms.single import GlobalBestPSO
import numpy as np
import time
import os

start = time.time()
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
n_dimensions = 5000
x_max = 5.12 * np.ones(n_dimensions)
x_min = - x_max
bounds = (x_min, x_max)
iterations = 5000
n_particles = 4000

optimizer = GlobalBestPSO(n_particles= n_particles, dimensions = n_dimensions,options = options, bounds= bounds)
optimizer.optimize(rastrigin,iters = iterations)

cost_history = optimizer.cost_history
position_history = optimizer.pos_history[0]

end = time.time()
function_evaluations = iterations*n_particles

# optimization_time = np.linspace(0, end - start, num= len(cost_history))

optimization_time = np.linspace(0, function_evaluations, num= len(cost_history))
print(f'The PSO algorithm took {end-start} seconds')


results_path = 'Results'
np.savez(os.path.join(results_path,"testing_pso.npz"), cost_history,position_history,optimization_time ,cost_h = cost_history,
         time = optimization_time, pos_history = position_history)
