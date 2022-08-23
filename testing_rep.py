import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import cm
import os
import matplotlib
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import latex
# DATA IMPORT FROM NPZ FILES
results_path = 'Results'
optimization_results = np.load(os.path.join(results_path,'q10.npz'))
Q_history = optimization_results['pos_history']
cost_history = optimization_results['cost_h']
history = optimization_results['time']
#
# optimization_results_2 = np.load(os.path.join(results_path,'testing_pso.npz'))
# cost_history_2 = optimization_results_2['cost_h']
# history_2 = optimization_results_2['time']
#
# optimization_results_3 = np.load(os.path.join(results_path,'qea_testing.npz'))
# Q_history_3 = optimization_results_3['pos_history']
# cost_history_3 = optimization_results_3['cost_h']
# history_3 = optimization_results_3['time']
#
# optimization_results_4 = np.load(os.path.join(results_path,'qea_testing0002.npz'))
# Q_history_4 = optimization_results_4['pos_history']
# cost_history_4 = optimization_results_4['cost_h']
# history_4 = optimization_results_4['time']
#
# optimization_results_5 = np.load(os.path.join(results_path,'testing_genetic_500.npz'))
# # Q_history_5 = optimization_results_5['pos_history']
# cost_history_5 = optimization_results_5['cost_h']
# history_5 = optimization_results_5['eval']

#
#
#FIRST FIGURE -> LOGARITHMIC COST EVOLUTION WRT TIME(SECONDS)

fig = plt.figure(figsize=(8,4))
# fig.suptitle(r'Logarithmic Convergence of the optimizer')


ax = plt.subplot(1,1, 1,xlim = [0, 20_000_000],xticks=[0, 1_000_000, 2_000_000, 15_000_000],
    xticklabels=["0", "1M", "2M", "15M"] ,ylim=[1e-0, 10000])
plt.yscale('log')

ax.plot(history[:,None], cost_history, label = 'N-QEA',color = "#CC5DE8")
# ax.plot(history_2, cost_history_2, label ='PSO', color = "#82C91E")
# ax.plot(history_5, cost_history_5, label ='Genetic', color = "#FF7043")
ax.set_ylabel("Log", ha="center", weight="bold")
ax.set_xlabel("Function evaluations", va="top")
ax.set_title("$F_2^{10000}$ optimization")
ax.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax.grid(True, "major", color="0.65", linewidth=0.85, zorder=-10)
# ax.tick_params(which="both", labelleft=False, left=False)
# ax.text(500_000, 1e-14, "$\\rho_\sigma =1.0001 $", color="#CC5DE8", zorder = -30)
ax.legend(edgecolor="None")

plt.show()