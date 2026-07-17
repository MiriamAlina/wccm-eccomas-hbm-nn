import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hbm_nn.nn_nonlinearity import compute_autodiff_jacobian
from hbm_nn.aft_nonlinearity import compute_analytical_jacobian
from hbm_nn.plotting import plot_jacobian_comparison


jac_info = pd.read_csv('data/cond_Om_force80_kt10000000_muN106.csv')
print('Frobenius norm error average:',
      np.round(np.average(jac_info['fnorm_error']) * 100, 3), '%')

# Import input data -----------------------------------------------------------
q_frc = pd.read_csv('data/frc_inputs_force80_kt10000000_muN106.csv')
q_frc = q_frc[['a1p', 'a3p', 'b3p']].to_numpy()
q_frc_full = np.hstack([
    np.zeros((q_frc.shape[0], 1)),
    q_frc[:, [0]],
    np.zeros((q_frc.shape[0], 3)),
    q_frc[:, [1]],
    q_frc[:, [2]],
])

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
J_ana = np.empty((K, 4, 3))
for k in range(K):
    J_nn[k] = compute_autodiff_jacobian(nn_id, q_frc[k])

    # Elastic dry friction Jacobian
    J_ana_tmp = compute_analytical_jacobian(N, H, q_frc_full[k], k_t, f_c)
    J_ana[k, 0:2, 0] = J_ana_tmp[1:3, 1]
    J_ana[k, 0:2, 1:] = J_ana_tmp[1:3, 5:]
    J_ana[k, 2:, 0] = J_ana_tmp[5:, 1]
    J_ana[k, 2:, 1:] = J_ana_tmp[5:, 5:]

plot_jacobian_comparison(q_frc, J_nn, J_ana, figure_name='jenkins_jacobian',
                         file_format='png', save_figure=False)
plt.show()
