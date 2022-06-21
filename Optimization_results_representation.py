import numpy as np
from matplotlib import pyplot as plt
import os

# DATA IMPORT FROM NPZ FILES
results_path = 'Results'
optimization_results = np.load(os.path.join(results_path,'testing_evl.npz'))
Q_history = optimization_results['pos_history']
cost_history = optimization_results['cost_h']
history = optimization_results['time']

optimization_results_2 = np.load(os.path.join(results_path,'testing_evl.npz'))
cost_history_2 = optimization_results_2['cost_h']
history_2 = optimization_results_2['time']

#FIRST FIGURE -> LOGARITHMIC COST EVOLUTION WRT TIME(SECONDS)

fig = plt.figure(figsize=(6,6))
fig.suptitle('Logarithmic Convergence of the optimizer')
ax = plt.subplot(xlim = [0, 5000000])
ax.grid(True)
ax.plot(history[:,None], cost_history, label = 'elitism')
ax.plot(history_2, cost_history_2, label ='pso')
ax.set_ylabel("Cost", ha="center", weight="bold")
ax.set_xlabel("Function evaluations", va="center", weight="bold")
ax.legend(edgecolor="None")
plt.yscale('log')
plt.show()



# Q EVOLUTION OF THE 10 FIRST FEATURES OF THE QEA ALGORITHM.

fig_1 = plt.figure(figsize=(8, 8))
fig_1.suptitle('Features distribution convergence')

for i in range(10):
    mu_evolution = Q_history[:, 0, i]
    std_evolution = Q_history[:, 1, i]
    ax_1 = plt.subplot(2, 5, i+1)
    ax_1.set_ylim(-1,1)
    ax_1.set_title(f'Feature {i}')
    ax_1.fill_between(history,mu_evolution + std_evolution, mu_evolution - std_evolution, facecolor="C0", alpha=0.25, zorder=-40)
    ax_1.plot(history, mu_evolution, color="C0", zorder=-30)
#ax_1.set_xticks([]), ax_1.set_yticks([])

plt.show()



print('process ended')