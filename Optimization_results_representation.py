import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import cm
import os

# DATA IMPORT FROM NPZ FILES
results_path = 'Results'
optimization_results = np.load(os.path.join(results_path,'testing_evl2.npz'))
Q_history = optimization_results['pos_history']
cost_history = optimization_results['cost_h']
history = optimization_results['time']

optimization_results_2 = np.load(os.path.join(results_path,'testing_evl_top_2.npz'))
cost_history_2 = optimization_results_2['cost_h']
history_2 = optimization_results_2['time']

#FIRST FIGURE -> LOGARITHMIC COST EVOLUTION WRT TIME(SECONDS)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Logarithmic Convergence of the optimizer')
ax = plt.subplot(xlim = [0, 15000000])
ax.grid(True)
# ax.plot(history[:,None], cost_history, label = 'elitism')
ax.plot(history_2, cost_history_2, label ='pso')
ax.set_ylabel("Cost", ha="center", weight="bold")
ax.set_xlabel("Function evaluations", va="center", weight="bold")
ax.legend(edgecolor="None")
plt.yscale('log')
plt.show()



# Q EVOLUTION OF THE 10 FIRST FEATURES OF THE QEA ALGORITHM.

fig_1 = plt.figure(figsize=(8, 8))
fig_1.suptitle('Features distribution convergence')
j = 0
for i in range(2):
    mu_evolution = Q_history[:, 0, i]
    std_evolution = Q_history[:, 1, i]
    ax_1 = plt.subplot(1, 2, j+1)
    ax_1.set_ylim(-1,1)
    ax_1.set_title(f'Feature {i}')
    ax_1.fill_between(history,mu_evolution + std_evolution, mu_evolution - std_evolution, facecolor="C0", alpha=0.25, zorder=-40)
    ax_1.plot(history, mu_evolution, color="C0", zorder=-30)
    j+=1
#ax_1.set_xticks([]), ax_1.set_yticks([])

plt.show()

#Explanation part to paint.

#Step 1: Paint rastrigin function
from test_functions import rastrigin


def objective(X, Y):
    Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
         (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return Z


# define range for input
r_min, r_max = -5.2, 5.2
# sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.05)
yaxis = np.arange(r_min, r_max, 0.05)
# create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)



fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.contourf(x, y, results, levels=250, cmap='rainbow',alpha = 0.5)
axs.set_xlim(-5,5)
axs.set_ylim(-5,5)
# define the known function optima
optima_x = [0.0, 0.0]
# draw the function optima as a white star
#plt.plot([optima_x[0]], [optima_x[1]], '*', color='white')
# show the plot
N=int(Q_history.shape[0]/2)

color = iter(cm.plasma(np.linspace(0, 1, N)))
for i in range(N):
    c = next(color)
    elipse = Ellipse((Q_history[i,0,0],Q_history[i,0,1]), 2*Q_history[i,1,0], 2*Q_history[i,1,1],
                     facecolor='None',edgecolor=c,alpha = 0.4)
    axs.add_patch(elipse)
axs.scatter(Q_history[:,0,0],Q_history[:,0,1],color='violet',s=0.6)
plt.show()

