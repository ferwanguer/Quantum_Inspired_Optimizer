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
optimization_results = np.load(os.path.join(results_path,'Ackley__big_2.npz'))
Q_history = optimization_results['pos_history']
cost_history = optimization_results['cost_h']
history = optimization_results['time']

#FIRST FIGURE -> LOGARITHMIC COST EVOLUTION WRT TIME(SECONDS)

fig = plt.figure(figsize=(8,4))
# fig.suptitle(r'Logarithmic Convergence of the optimizer')


ax = plt.subplot(1,1, 1,xlim = [0, 100_000_000],xticks=[0, 5_000_000, 10_000_000, 35_000_000],
    xticklabels=["0", "5M", "10M", "12M"] ,ylim=[1e-7, 1e7])
plt.yscale('log')

ax.plot(history[:,None], cost_history, label = 'N-QEA',color = "#CC5DE8")
# ax.plot(history_2, cost_history_2, label ='PSO', color = "#82C91E")
# ax.plot(history_5, cost_history_5, label ='Genetic', color = "#FF7043")
ax.set_ylabel("Log", ha="center", weight="bold")
ax.set_xlabel("Function evaluations", va="top")
ax.set_title("$F_2^{100000}$ optimization")
ax.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax.grid(True, "major", color="0.65", linewidth=0.85, zorder=-10)
# ax.tick_params(which="both", labelleft=False, left=False)
# ax.text(500_000, 1e-14, "$\\rho_\sigma =1.0001 $", color="#CC5DE8", zorder = -30)
ax.legend(edgecolor="None")

plt.show()