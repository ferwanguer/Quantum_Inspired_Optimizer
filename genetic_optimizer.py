import numpy as np
from test_functions import f, g, michael, rastrigin, rosenbrock, schwefel
from geneticalgorithm import geneticalgorithm as ga

varbound=np.array([[0,np.pi]]*2)
# Ojo con con el cambio que he hecho en la librería. Repasar mañana.

model=ga(function=michael,dimension=2,variable_type='real',variable_boundaries=varbound)

model.run()
# Ok 