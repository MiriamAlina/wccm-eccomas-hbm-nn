import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
from hbm_nn.artifact_selection import select_data_id
from hbm_nn.plotting import plot_inputs_3d, plot_inputs_pairwise


# FRC inputs ------------------------------------------------------------------
q_frc = pd.read_csv('data/frc_inputs_force80_kt10000000_muN106.csv')
q_frc = q_frc[['a1p', 'a3p', 'b3p']].to_numpy()

# Training data ---------------------------------------------------------------
data_id = select_data_id()
data_path = f'data/train_data_H3_{data_id}.npz'
data = np.load(data_path)
q_train = data['q_coeffs']  # input: [a1, a3, b3]
a1_train = q_train[:, 0]
a3_train = q_train[:, 1]
b3_train = q_train[:, 2]

plot_inputs_3d(q_train, q_frc, figure_name='training_samples_vs_frc_inputs_3d',
               file_format='png', save_figure=False)

plot_inputs_pairwise(q_train, q_frc,
                     figure_name='training_samples_vs_frc_inputs_pairwise',
                     file_format='png', save_figure=False)

plt.show()
