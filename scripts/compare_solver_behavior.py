import pandas as pd
import matplotlib.pyplot as plt
from hbm_nn.plotting import plot_solver_behavior


# Import data -----------------------------------------------------------------
ref = pd.read_csv(
    'data/aft_results_force80_kt10000000_muN106.csv')
test = pd.read_csv('data/nn_results_force80_kt10000000_muN106.csv')
cond = pd.read_csv('data/cond_Om_force80_kt10000000_muN106.csv')

# Plot results ----------------------------------------------------------------
Om_stick = 1.3410e+03  # normalize frequency axis where necessary
plot_solver_behavior(
    frequencies=[ref['freq_ratio'], test['freq_ratio']],
    amplitudes=[ref['Q_nl'], test['Q_nl']],
    iter_num=[ref['NIT'], test['NIT']],
    cond_frequencies=cond['Om']/Om_stick,
    cond_numbers=[cond['cond_AFT'], cond['cond_NN']],
    labels=['AFT', 'NN'],
    figure_name='EB1to3_FRC_comparison_with_inset',
    file_format='pdf',
    save_figure=False
)

plt.show()
