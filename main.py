import numpy as np
from qea import QuantumEvAlgorithm
import time
import os
"""This script defines the functions that may be optimized"""


def f(x: np.ndarray):
    """n-dimensional paraboloid definition. For the first test of the optimization algorithm."""
    n_dims = x.shape[1]
    min_value = np.random.rand(1, n_dims)
    f = np.sum(np.square(x - min_value), axis=1)[:, None]

    return f


def g(x: np.ndarray):

    """n-dimensional Ackley function definition. For the second test of the optimization algorithm. """
    n_dims = x.shape[1]

    a = 20.0
    b = 0.2
    c = 2 * np.pi

    auxiliar = -b * np.sqrt(np.divide(np.sum(np.square(x), axis=1), n_dims))
    auxiliar_2 = np.sum(np.cos(c * x), axis=1) / n_dims
    g = -a * np.exp(auxiliar) - np.exp(auxiliar_2) + a + np.exp(1.0)

    return g

def rastrigin(x: np.ndarray):
    """n-dimensional rastrigin function, testing purposes"""
    n_dims = x.shape[1]
    a = 10
    x_squared = np.square(x)
    x_sin = a * np.cos(2 * np.pi * x)
    rastrigin = a * n_dims + np.sum(x_squared - x_sin, axis = 1)

    return rastrigin

def rosenbrock(x: np.ndarray):
    x_plus = x[:,1:]
    x_i = x[:,0:-1]
    aux_1 = 100 * np.square(x_plus - np.square(x_i))
    aux_2 = np.square(x_i - 1)
    rosenbrock = np.sum(aux_1 + aux_2, axis = 1)
    return rosenbrock


# _,min_solution = f(np.random.rand(1,10))

if __name__ == '__main__':
    n_dims = 500
    optimizer = QuantumEvAlgorithm(rosenbrock, n_dims = n_dims) # Instatiation of the class with the ackley function
    Q = optimizer.quantum_individual_init() #Quantum individual initialization

    N_iterations = 8000000
    sample_size = 10
    saving_interval = 50 # Every 500 iterations the cost history is saved

    Q_history = np.zeros((int(N_iterations / saving_interval), 2, n_dims))
    best_performer_marker = np.zeros((int(N_iterations / saving_interval), 1))

    j = 0
    print('Beggining of the iteration process')
    beginning = time.time()

    for i in range(N_iterations):
        samples = optimizer.quantum_sampling(Q, sample_size)
        best_performer = optimizer.elitist_sample_evaluation(samples)
        Q = optimizer.quantum_update(Q, best_performer,i)
        if np.mod(i, saving_interval) == 0:
            Q_history[j, :, :] = Q

            output = rosenbrock(best_performer)
            # print(f'Current value of ackley is: {output}')
            best_performer_marker[j, :] = output
            j += 1
        if np.mod(i, 50000) == 0:
            print(f'Iteration {i}, Best cost = {output}')
    end = time.time()

    print(f'The algorithm took {end - beginning} seconds')
    print(f' min is  = {rosenbrock(best_performer)}')
    print(f'The min is IN = {best_performer}')

    function_evaluations: int = sample_size * N_iterations
    # optimization_time = np.linspace(0, end - beginning, num=len(best_performer_marker))
    optimization_time = np.linspace(0, function_evaluations, num=len(best_performer_marker))
    # Saving the results. We save into a npz file the cost history, the feature history and an optimization
    # time vector. Optimization results representation

    results_path = 'Results'
    # np.savez(os.path.join(results_path,"testing_ev.npz"), best_performer_marker, Q_history, optimization_time,
    #          cost_h=best_performer_marker,
    #          pos_history=Q_history, time=optimization_time)

    print('End')
