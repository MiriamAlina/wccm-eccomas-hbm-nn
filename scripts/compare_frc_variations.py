import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hbm_nn.plotting import plot_frc_variations
from hbm_nn.util import check_folder_structure


check_folder_structure()


def load_results(force, kt, muN):
    ref = pd.read_csv(
        f'./data/aft_results_force{force}_kt{kt}_muN{muN}.csv')
    test = pd.read_csv(
        f'./data/nn_results_force{force}_kt{kt}_muN{muN}.csv')
    frequencies = [ref['freq_ratio'].to_numpy(), test['freq_ratio'].to_numpy()]
    amplitudes = [ref['Q_nl'].to_numpy(), test['Q_nl'].to_numpy()]
    return frequencies, amplitudes


# System Parameters -----------------------------------------------------------
force_list = [45, 75, 105]
fixed_force = 45
kt_list = [int(1e7), int(1e7*.5), int(1e7*.35)]
fixed_kt = int(1e7)
muN_list = [int(.5*53*4 * .5), int(.5*53*4), int(.5*53*4 * 5)]
fixed_muN = int(.5*53*4)
method_labels = ['AFT', 'NN']

# Load Results ----------------------------------------------------------------
freqs_by_force = []
amps_by_force = []
for force in force_list:
    freqs, amps = load_results(force, fixed_kt, fixed_muN)
    freqs_by_force.append(freqs)
    amps_by_force.append(amps)

freqs_by_muN = []
amps_by_muN = []
for muN in muN_list:
    freqs, amps = load_results(fixed_force, fixed_kt, muN)
    freqs_by_muN.append(freqs)
    amps_by_muN.append(amps)

freqs_by_kt = []
amps_by_kt = []
for kt in kt_list:
    freqs, amps = load_results(fixed_force, kt, fixed_muN)
    freqs_by_kt.append(freqs)
    amps_by_kt.append(amps)

# Plot Results ----------------------------------------------------------------
plot_frc_variations(
    data_list=[kt_list, muN_list, force_list],
    freqs=[freqs_by_kt, freqs_by_muN, freqs_by_force],
    amps=[amps_by_kt, amps_by_muN, amps_by_force],
    variation_value_labels=[
        [rf'$k_t={kt / 10**int(np.floor(np.log10(abs(kt)))):.2g}'
         rf'\cdot10^{{{int(np.floor(np.log10(abs(kt))))}}}$'
         for kt in kt_list],
        [rf'$\mu F_N={muN:.0f}$' for muN in muN_list],
        [rf'$F_0={force:.0f}$' for force in force_list],
    ],
    fixed_params_labels=[
        r'fixed $F_0=45$' + '\n' + r'fixed $\mu F_N=106$',
        r'fixed $F_0=45$' + '\n' + r'fixed $k_t=10^7$',
        r'fixed $k_t=10^7$' + '\n' + r'fixed $\mu F_N=106$',
    ],
    colors=['#457b9d', '#e63946', 'k'],
    save_figure=False,
    figure_name='frc_variations',
    file_format='pdf'
)

plt.show()
