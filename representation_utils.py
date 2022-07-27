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
optimization_results = np.load(os.path.join(results_path,'qea_testing_sigma_0001.npz'))
Q_history = optimization_results['pos_history']
cost_history = optimization_results['cost_h']
history = optimization_results['time']

optimization_results_2 = np.load(os.path.join(results_path,'testing_pso.npz'))
cost_history_2 = optimization_results_2['cost_h']
history_2 = optimization_results_2['time']

optimization_results_3 = np.load(os.path.join(results_path,'qea_testingm.npz'))
Q_history_3 = optimization_results_3['pos_history']
cost_history_3 = optimization_results_3['cost_h']
history_3 = optimization_results_3['time']

optimization_results_4 = np.load(os.path.join(results_path,'qea_testing0002.npz'))
Q_history_4 = optimization_results_4['pos_history']
cost_history_4 = optimization_results_4['cost_h']
history_4 = optimization_results_4['time']



#FIRST FIGURE -> LOGARITHMIC COST EVOLUTION WRT TIME(SECONDS)

fig = plt.figure(figsize=(8,5))
# fig.suptitle(r'Logarithmic Convergence of the optimizer')


# ax = plt.subplot(1,3, 3,xlim = [0, 2_000_000],xticks=[0, 1_000_000, 2_000_000],
#     xticklabels=["0", "1M", "2M"] ,ylim=[1e-15, 100])
# plt.yscale('log')
#
# ax.plot(history[:,None], cost_history, label = 'N-QEA',color = "#CC5DE8")
# ax.plot(history_2, cost_history_2, label ='PSO', color = "#82C91E")
# # ax.set_ylabel("Logarithmic cost", ha="center", weight="bold")
# ax.set_xlabel("Function evaluations", va="top")
# ax.set_title("(c)")
# ax.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
# ax.grid(True, "major", color="0.65", linewidth=0.85, zorder=-10)
# ax.tick_params(which="both", labelleft=False, left=False)
# ax.text(500_000, 1e-14, "$\\rho_\sigma =1.0001 $", color="#CC5DE8", zorder = -30)
# ax.legend(edgecolor="None")


##############################################################################

ax1 = plt.subplot(1,1, 1,xlim = [0, 1_000_000],xticks=[0, 500_000, 1_000_000],
    xticklabels=["0", "500K", "1M"] , ylim=[1e-4, 100])
plt.yscale('log')
ax1.plot(history_3[:,None], cost_history_3, label = 'N-QEA',color = "#CC5DE8")
ax1.plot(history_2, cost_history_2, label ='PSO', color = "#82C91E")
ax1.set_ylabel("Log cost", ha="center", weight="bold")
ax1.set_xlabel("Function evaluations", va="top")
# ax1.set_title("(a)")
ax1.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax1.grid(True, "major", color="0.65", linewidth=0.85, zorder=-10)
ax1.text(250_000, 1e-14, "$\\rho_\sigma =1.001 $", color="#CC5DE8", zorder = -30)
ax1.legend(edgecolor="None")

########################################################################################
#
# ax2 = plt.subplot(1,3, 2,xlim = [0, 2_000_000],xticks=[0, 1_000_000, 2_000_000],
#     xticklabels=["0", "1M", "2M"] , ylim=[1e-15, 100])
# plt.yscale('log')
# ax2.plot(history_4[:,None], cost_history_4, label = 'N-QEA',color = "#CC5DE8")
# ax2.plot(history_2, cost_history_2, label ='PSO', color = "#82C91E")
# ax2.set_title("(b)")
# #ax2.set_ylabel("Logarithmic cost", ha="center", weight="bold")
# ax2.set_xlabel("Function evaluations", va="top")
# ax2.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
# ax2.grid(True, "major", color="0.65", linewidth=0.85, zorder=-10)
# ax2.tick_params(which="both", labelleft=False, left=False)
# ax2.text(800_000, 1e-14, "$\\rho_\sigma =1.0002 $", color="#CC5DE8", zorder = -30)
# ax2.legend(edgecolor="None")
#
#
#
# axins = zoomed_inset_axes(ax1, zoom = 150, loc="center right")
# axins.plot(history_3[:,None], cost_history_3,'-',color = "#CC5DE8")
# axins.set_xlim(50000, 60000)
# axins.set_xticks([])
# axins.set_ylim(1e-5, 6e-5)
# # axins.set_adjustable("box")
# axins.set_yticks([])
# axins.set_aspect(100000000)
#
#
# rect = Rectangle(
#     (100, 4e-6),
#     200_000,
#     2e-5,
#     edgecolor="black",
#     facecolor="None",
#     linestyle="--",
#     linewidth=0.75,
# )
# ax1.add_patch(rect)
#
# con = ConnectionPatch(
#     xyA=(200_000, 1e-5),
#     coordsA=ax1.transData,
#     xyB=(0.22, 0.55),
#     coordsB=ax1.transAxes,
#     linestyle="--",
#     linewidth=0.75,
#     patchA=rect,
#     arrowstyle="->",
# )
# fig.add_artist(con)

plt.show()

# Q EVOLUTION OF THE 10 FIRST FEATURES OF THE QEA ALGORITHM.

fig_1 = plt.figure(figsize=(8, 3))
# fig_1.suptitle('Features distribution convergence')
j = 0
N=7
for i in range(N):
    mu_evolution = Q_history_3[:, 0, i]
    std_evolution = Q_history_3[:, 1, i]
    ax_1 = plt.subplot(1, N, j+1)
    ax_1.set_ylim(-10,10)
    ax_1.set_xlim(0,20000)
    ax_1.set_title(f'Feature {i}')
    if i == 0:
        ax_1.set_yticks([-10,-5,0,5,10])
    else :
        ax_1.set_yticks([])
    ax_1.set_xticks([20000])
    ax_1.fill_between(history_3,mu_evolution + std_evolution, mu_evolution - std_evolution, facecolor="C0", alpha=0.25, zorder=-40)
    ax_1.plot(history_3, mu_evolution, color="C0", zorder=-30)
    j+=1
#ax_1.set_xticks([]), ax_1.set_yticks([])

plt.show()

#Explanation part to paint.

#Step 1: Paint rastrigin function
from test_functions import rastrigin
#
#
# def objective(X, Y):
#     Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
#          (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
#     return Z
#
#
# # define range for input
# r_min, r_max = -5.2, 5.2
# # sample input range uniformly at 0.1 increments
# xaxis = np.arange(r_min, r_max, 0.05)
# yaxis = np.arange(r_min, r_max, 0.05)
# # create a mesh from the axis
# x, y = np.meshgrid(xaxis, yaxis)
# # compute targets
# results = objective(x, y)


#
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# fig.tight_layout()
# axs.contourf(x, y, results, levels=250, cmap='rainbow',alpha = 0.7)
# axs.set_xlim(-5,5)
# axs.set_ylim(-5,5)
# # define the known function optima
# optima_x = [0.0, 0.0]
# # draw the function optima as a white star
# #plt.plot([optima_x[0]], [optima_x[1]], '*', color='white')
# # show the plot
# N=int(Q_history.shape[0]/3)
#
# color = iter(cm.plasma(np.linspace(1, 0, N)))
# for i in range(N):
#     c = next(color)
#     elipse = Ellipse((Q_history[i,0,0],Q_history[i,0,1]), 2*Q_history[i,1,0], 2*Q_history[i,1,1],
#                      facecolor=c,edgecolor=c,alpha = 0.15)
#     axs.add_patch(elipse)
# axs.scatter(Q_history[:,0,0],Q_history[:,0,1],color='violet',s=1.8)
# axs.set_xticks([])
# axs.set_yticks([])
# norm = matplotlib.colors.Normalize(vmin=0, vmax=5)
# plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.plasma))
# plt.show()

