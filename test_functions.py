import numpy as np

"""This script defines the functions that are to be optimized"""


def f(x: np.ndarray):
    """n-dimensional paraboloid definition. For the first test of the optimization algorithm."""
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    min_value = 0.8 * np.ones((1, n_dims))
    f = np.sum(np.square(x - min_value), axis=1)

    return f


def g(x: np.ndarray):

    """n-dimensional Ackley function definition. For the second test of the optimization algorithm. """
    if x.ndim == 1:
        x = x[None]

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
    if x.ndim == 1:
        x = x[None]

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
