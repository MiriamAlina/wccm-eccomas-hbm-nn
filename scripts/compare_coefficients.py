import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hbm_nn.aft_nonlinearity import compute_nonlinear_force_coefficients
from hbm_nn.fourier_conversion import (convert_cossin_to_comexp,
                                       convert_comexp_to_cossin)
from hbm_nn.artifact_selection import select_model_id
from hbm_nn.nn_nonlinearity import infer_nonlinear_force_coefficients
from hbm_nn.error_metrics import compute_error_metrics
from hbm_nn.plotting import (
    plot_coefficients_over_iterations,
    plot_prediction_vs_ground_truth_with_inset,
    individual_normalized_mse_bar_plot,
    spider_plot_error_metrics
)
from hbm_nn.util import check_folder_structure

os.environ["CUDA_VISIBLE_DEVICES"]=""

check_folder_structure()

# Performance on FRC trajectory -----------------------------------------------
q_frc = pd.read_csv('data/frc_inputs_force80_kt10000000_muN106.csv')
q_frc = q_frc[['a1p', 'a3p', 'b3p']].to_numpy()
q_frc_full = np.hstack([
    np.zeros((q_frc.shape[0], 1)),
    q_frc[:, [0]],
    np.zeros((q_frc.shape[0], 3)),
    q_frc[:, [1]],
    q_frc[:, [2]],
])

nn_id = select_model_id()

H = 3
N = 2**6
kt = 1.0  # Example value, replace with actual
fc = 1.0  # Example value, replace with actual
fnl_rel_aft = np.empty((0, 4))
fnl_rel_nn = np.empty((0, 4))
for i in range(np.shape(q_frc_full)[0]):
    q_ce = convert_cossin_to_comexp(q_frc_full[i])
    fnl_ce = compute_nonlinear_force_coefficients(N, H, q_ce, kt, fc)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)
    fnl_rel_aft = np.vstack([fnl_rel_aft, fnl_cs[[1, 2, 5, 6]]])

    fnl_cs_NN = infer_nonlinear_force_coefficients(nn_id, q_frc[i])
    fnl_rel_nn = np.vstack([fnl_rel_nn, fnl_cs_NN])

# Compute error metrics and plot results --------------------------------------
global_metrics_frc, individual_metrics_frc = \
    compute_error_metrics(fnl_rel_aft, fnl_rel_nn)
global_metrics_frc_normalized, individual_metrics_frc_normalized = \
    compute_error_metrics(fnl_rel_aft, fnl_rel_nn, normalize=True)

print('Relative L2 norm error average:',
      np.round(global_metrics_frc['Relative\nL$^2$ Norm'] * 100, 3), '%')

plot_prediction_vs_ground_truth_with_inset(
    [fnl_rel_aft],
    [fnl_rel_nn],
    figure_name='predictions_vs_ground_truths', file_format='pdf',
    save_figure=False)

plot_coefficients_over_iterations(
    np.hstack([q_frc[:, 0:1], np.zeros((q_frc.shape[0], 1)), q_frc[:, 1:3]]),
    fnl_rel_aft,
    fnl_rel_nn)

individual_normalized_mse_bar_plot(individual_metrics_frc_normalized,
                                   figure_name='nmse_bar_frc',
                                   file_format='pdf', save_figure=False)

spider_plot_error_metrics(global_metrics_frc, global_metrics_frc_normalized,
                          figure_name='error_metrics_spider',
                          file_format='pdf', save_figure=False)

plt.show()
