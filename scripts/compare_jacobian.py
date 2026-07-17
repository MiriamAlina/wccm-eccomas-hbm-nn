import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from hbm_nn.nn_nonlinearity import evaluate_nn_model
from hbm_nn.aft_nonlinearity import get_analytical_jacobian


jac_info = pd.read_csv('data/EB1to3_cond_Om_force80_kt10000000_muN106.csv')

print('fnorm_error average:', np.average(jac_info['fnorm_error']))


# Import data -----------------------------------------------------------------
data = pd.read_csv(
    'data/EB1to3_cond_Om_input_jac_force80_kt10000000_muN106.txt',
    header=None
)
Om = data.iloc[:, 2]
nn_input = data.iloc[:, 3:3+3]  # 3 columns
nn_input = np.reshape(nn_input.values, (nn_input.shape[0], 3))
q_frc = nn_input  # Only take the first 3 columns as input
q_frc_full = np.hstack([np.zeros((q_frc.shape[0], 1)), q_frc[:, 0:1],
                        np.zeros((q_frc.shape[0], 3)), q_frc[:, 1:3]])

# Load NN ---------------------------------------------------------------------
nn_id = '2026-04-16_09-31-57'  # '2026-04-01_11-16-24'
nn_path = f'models/MLP_Jenkins_H3_{nn_id}.pt'
k_t = 1.0
f_c = 1.0
K = q_frc_full.shape[0]
H = 3
N = 4*H+1

# Compute Jacobians -----------------------------------------------------------
J_nn = np.empty((K, 4, 3))
J_nn_full = np.empty((K, 7, 7))
J_edf = np.empty((K, 7, 7))
for k in range(K):
    J_nn[k] = evaluate_nn_model(nn_id, nn_input[k])
    J_nn_full[k, 1:3, 1:3] = J_nn[k, 0:2, 0:2]
    J_nn_full[k, 1:3, 5:] = J_nn[k, 0:2, 2:]
    J_nn_full[k, 5:, 1:3] = J_nn[k, 2:, 0:2]
    J_nn_full[k, 5:, 5:] = J_nn[k, 2:, 2:]

    # Elastic dry friction Jacobian
    _, J_edf[k] = get_analytical_jacobian(N, H, q_frc_full[k], k_t, f_c)

J_edf_sub = np.empty((K, 4, 3))
J_edf_sub[:, 0:2, 0] = J_edf[:, 1:3, 1]
J_edf_sub[:, 0:2, 1:] = J_edf[:, 1:3, 5:]
J_edf_sub[:, 2:, 0] = J_edf[:, 5:, 1]
J_edf_sub[:, 2:, 1:] = J_edf[:, 5:, 5:]

PLOT_0 = True
if PLOT_0:
    input_labels = [r"$a_0$", r"$a_1$", r"$b_1$", r"$a_2$",
                    r"$b_2$", r"$a_3$", r"$b_3$"]
    output_labels = [r"$A_0$", r"$A_1$", r"$B_1$", r"$A_2$",
                     r"$B_2$", r"$A_3$", r"$B_3$"]

    fig, axes = plt.subplots(7, 7, figsize=(14, 8), sharex="col")

    for i in range(7):
        for j in range(7):
            ax = axes[i, j]
            x = q_frc_full[:, j]      # x_j
            y_nn = J_nn_full[:, i, j]      # NN: J_ij
            y_edf = J_edf[:, i, j]      # EDF: J_ij
            ax.scatter(x, y_edf, s=6, alpha=0.35, color='#A8DADC',
                       label="EDF" if (i == 0 and j == 0) else None)
            ax.scatter(x, y_nn, s=6, alpha=0.35, color='#E63946',
                       label="NN" if (i == 0 and j == 0) else None)

            if i == 2:
                ax.set_xlabel(input_labels[j])
            if j == 0:
                ax.set_ylabel(output_labels[i])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("FD vs NN: Jacobian entry J[i,j] plotted against input x_j")
    plt.tight_layout()


PLOT_1 = True
if PLOT_1:
    input_labels = [r"$a_1$", r"$a_3$", r"$b_3$"]
    output_labels = [r"$A_1$", r"$B_1$", r"$A_3$", r"$B_3$"]

    input_symbols = [r"a_1", r"a_3", r"b_3"]
    output_symbols = [r"A_1", r"B_1", r"A_3", r"B_3"]

    def zero_clean_formatter(x, pos):
        if abs(x) < 1e-12:
            return "0"
        return f"{x:g}"

    fig, axes = plt.subplots(4, 3, figsize=(10, 7), sharex="col")
    for i in range(4):  # output index (row)
        for j in range(3):  # input index (col)
            ax = axes[i, j]
            x = q_frc[:, j]
            y_nn = J_nn[:, i, j]
            y_edf = J_edf_sub[:, i, j]
            ax.scatter(x, y_edf, s=2, alpha=0.4, color='k',
                       label="Finite Differences" if (i == 0 and j == 0)
                       else None)
            ax.scatter(x, y_nn, s=2, alpha=0.4, color='lightgrey',
                       label="Neural Network" if (i == 0 and j == 0)
                       else None)
            if i == 3:
                ax.set_xlabel(input_labels[j], fontsize=12)

            ylabel = (fr"$\frac{{\partial {output_symbols[i]}}}"
                      fr"{{\partial {input_symbols[j]}}}$")
            ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=15,
                          va="center")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)
    fig.subplots_adjust(wspace=0.8, hspace=0.45)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure = 0
    if save_figure:
        plt.savefig("figures/jenkins_jacobian.png", dpi=300,
                    bbox_inches="tight")
    plt.show()


###############################################################################
# Compare conditioning: smallest singular value and condition number
###############################################################################
smin_fd = np.empty(K)
smin_nn = np.empty(K)
cond_fd = np.empty(K)
cond_nn = np.empty(K)

for k in range(K):
    sv_fd = np.linalg.svd(J_edf_sub[k], compute_uv=False)
    sv_nn = np.linalg.svd(J_nn[k], compute_uv=False)
    smin_fd[k] = sv_fd[-1]
    smin_nn[k] = sv_nn[-1]
    cond_fd[k] = sv_fd[0] / max(sv_fd[-1], 1e-14)
    cond_nn[k] = sv_nn[0] / max(sv_nn[-1], 1e-14)
print('shape of smin arrays:', smin_fd.shape, smin_nn.shape)

print("Smallest singular values")
print("FD: min smin =", np.min(smin_fd))
print("NN: min smin =", np.min(smin_nn))

print("\nCondition numbers")
print("FD: max cond =", np.max(cond_fd))
print("NN: max cond =", np.max(cond_nn))


###############################################################################
# Conditioning vs Omega
###############################################################################
# Omega = X_full[:, -1]
Omega = np.arange(K)  # Use index as x-axis for plotting

plt.plot(smin_fd, label="FD", marker='o', linestyle='-', markersize=2,
         color='k')
plt.plot(smin_nn, label="NN", marker='o', linestyle='-', markersize=2,
         color='grey')
plt.xlabel("Index (k)")
plt.ylabel("Smallest singular value")
plt.title("Smallest singular value vs Index")
plt.legend()
plt.grid()
plt.show()

plt.plot(cond_fd, label="FD", marker='o', linestyle='-', markersize=2,
         color='k')
plt.plot(cond_nn, label="NN", marker='o', linestyle='-', markersize=2,
         color='grey')
plt.xlabel("Index (k)")
plt.ylabel("Condition number")
plt.title("Condition number vs Index")
plt.legend()
plt.grid()
plt.show()


###############################################################################
# Linearisation quality test
###############################################################################
def linearisation_error(J, R, dx):
    return np.linalg.norm(R + J @ dx) / max(np.linalg.norm(R), 1e-14)


lin_err_fd = []
lin_err_nn = []

for k in range(K):
    Jfd = J_edf_sub[k]
    Jnn = J_nn[k]
    dx = np.random.randn(3)
    dx /= np.linalg.norm(dx)
    dx *= 1e-6
    Rfd = Jfd @ dx
    Rnn = Jnn @ dx
    lin_err_fd.append(np.linalg.norm(Rfd))
    lin_err_nn.append(np.linalg.norm(Rnn))

print("\nLinearisation quality")
print("FD mean:", np.mean(lin_err_fd))
print("NN mean:", np.mean(lin_err_nn))


###############################################################################
# Jacobian smoothness
###############################################################################
smooth_fd = []
smooth_nn = []

for k in range(K-1):
    smooth_fd.append(np.linalg.norm(J_edf_sub[k+1] - J_edf_sub[k]))
    smooth_nn.append(np.linalg.norm(J_nn[k+1] - J_nn[k]))

print("\nJacobian smoothness")
print("FD mean change:", np.mean(smooth_fd))
print("NN mean change:", np.mean(smooth_nn))

plt.show()
