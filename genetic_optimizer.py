import numpy as np
from ttest_functions import f, g, michael, rastrigin, rosenbrock, schwefel, dropwave
from geneticalgorithm import geneticalgorithm as ga

varbound=np.array([[-5,5]]*2)
# Ojo con con el cambio que he hecho en la librería. Repasar mañana.

model=ga(function=dropwave,dimension=2,variable_type='real',variable_boundaries=varbound)

model.run()
