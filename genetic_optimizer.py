import numpy as np
from test_functions import f, g, rastrigin, rosenbrock
from geneticalgorithm import geneticalgorithm as ga

varbound=np.array([[-5.12,5.12]]*100)
# Ojo con con el cambio que he hecho en la librería. Repasar mañana.

model=ga(function=f,dimension=100,variable_type='real',variable_boundaries=varbound)

model.run()
#odio los algoritmos de optimizagión