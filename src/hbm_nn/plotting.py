import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D


# Latex text font
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif",
                     "font.serif": ["Computer Modern Roman"],
                     "font.size": 12
                     })

input_labels = [r"$a_1$", r"$b_1$", r"$a_3$", r"$b_3$"]
output_labels = [r"$A_1$", r"$B_1$", r"$A_3$", r"$B_3$"]
two_colors_set = ['#1D3557', '#e63946']
four_colors_set = ['#1D3557', '#008b9a', '#f19699', '#e63946']
grayscale_colors_set = ['#323232', '#646464', '#969696', '#C8C8C8']


def plot_coefficients_over_iterations(
        input_coeffs,
        aft_outputs,
        nn_outputs):
    """
    Create a line plot showing the evolution of coefficients over iterations.
    Inputs:
        input_coeffs: Array of input coefficients over iterations (numpy array)
        aft_outputs: Array of AFT output coefficients over iterations
            (numpy array)
        nn_outputs: Array of NN output coefficients over iterations
            (numpy array)
    """

    fig, ax = plt.subplots(4, 1, figsize=(5, 8))
    for i in range(4):
        ax[0].plot(input_coeffs[:, i], label=input_labels[i],
                   color=four_colors_set[i])
        ax[1].plot(aft_outputs[:, i], color=four_colors_set[i])
        ax[2].plot(nn_outputs[:, i], color=four_colors_set[i])
        ax[3].plot(aft_outputs[:, i] - nn_outputs[:, i],
                   label=output_labels[i], color=four_colors_set[i])
    ax[0].set_title('Input coefficients over iterations')
    ax[0].legend(loc="upper right")
    ax[1].set_title('AFT output coefficients over iterations')
    ax[2].set_title('NN output coefficients over iterations')
    ax[3].legend(loc="upper right")
    ax[3].set_title('Difference of AFT and NN outputs over iterations')
    plt.tight_layout()


def plot_prediction_vs_ground_truth_with_inset(
        ground_truth,
        prediction,
        figure_name,
        file_format='svg',
        save_figure=False):
    """
    Create a scatter plot comparing the neural network's predictions to the
    ground truth values, with an inset for zoomed-in view.
    Inputs:
        ground_truth: List of ground truth arrays (list of numpy arrays)
        prediction: List of prediction arrays (list of numpy arrays)
        figure_name: Name for saving the figure (str)
        file_format: Format for saving the figure (str, default='svg')
        save_figure: Flag to save the figure (bool, default=False)
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    fig.subplots_adjust(left=0.15, right=0.7, bottom=0.15, top=0.95)
    pt = (0, 0)
    ax.axline(pt, slope=1, color='black', linewidth=0.5)
    for i in range(4):
        for gt, pred in zip(ground_truth, prediction):
            ax.plot(gt[:, i], pred[:, i], '.',
                    label=f'{output_labels[i]}',
                    color=four_colors_set[i])
    ax.set_xlabel('AFT Ground Truth')
    ax.set_ylabel('Neural Network Prediction')
    ax.legend()

    axins = inset_axes(
        ax,
        width=1.2,
        height=1.2,
        loc="lower right",
        bbox_to_anchor=(1.45, 0.15),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    axins.axline(pt, slope=1, color='black')
    for i in range(4):
        for gt, pred in zip(ground_truth, prediction):
            axins.plot(gt[:, i], pred[:, i], '.',
                       color=four_colors_set[i])
    axins.set_xlim(-0.07, 0.07)
    axins.set_xticks([-0.05, 0., 0.05])
    axins.set_ylim(-0.07, 0.07)
    axins.set_yticks([-0.05, 0., 0.05])
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    if save_figure:
        plt.savefig(f'figures/{figure_name}.{file_format}',
                    bbox_inches='tight')


def individual_normalized_mse_bar_plot(
        normalized_metrics_dict,
        figure_name,
        file_format='svg',
        save_figure=False):
    """
    Create a bar plot for individual normalized mean squared error (MSE)
    metrics.
    Inputs:
        normalized_metrics_dict: Dictionary containing normalized MSE metrics
        figure_name: Name for saving the figure (str)
        file_format: Format for saving the figure (str, default='svg')
        save_figure: Flag to save the figure (bool, default=False)
    """
    norm = normalized_metrics_dict['MSE']
    x = np.arange(len(norm))
    w = 0.4
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    ax.bar(x, norm, label=output_labels, width=w, color=four_colors_set)
    ax.set_xticks(x)
    ax.set_xticklabels(output_labels)
    ax.set_xlabel('Neural Network Output')
    ax.set_ylabel("Normalized Mean Squared Error")
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figures/{figure_name}.{file_format}',
                    bbox_inches='tight')


def spider_plot_error_metrics(
        metrics_dict,
        normalized_metrics_dict,
        figure_name,
        file_format='svg',
        save_figure=False):
    """
    Create a spider plot comparing error metrics before and after
    normalization.
    Inputs:
        metrics_dict: Dictionary of error metrics before normalization
        normalized_metrics_dict: Dictionary of error metrics after
            normalization
        figure_name: Name of the figure file to save (str)
        file_format: Format for saving the figure, default is 'svg' (str)
        save_figure: Whether to save the figure, default is False (bool)
    """
    labels = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in labels]
    values_norm = [normalized_metrics_dict[k] for k in labels]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.r_[values, values[0]]
    values_norm = np.r_[values_norm, values_norm[0]]
    angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
    ax.set_yscale("log")
    ax.plot(angles, values, marker="o", label="raw", color=two_colors_set[0])
    ax.fill(angles, values, alpha=0.2, color=two_colors_set[0])
    ax.plot(angles, values_norm, marker="o", label="normalized",
            color=two_colors_set[1])
    ax.fill(angles, values_norm, alpha=0.2, color=two_colors_set[1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', pad=12)
    ax.set_ylim(min(values)/3, max(values)*6)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.4), ncol=2)
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figures/{figure_name}.{file_format}',
                    bbox_inches='tight')


def plot_frc_variations(data_list,
                        freqs,
                        amps,
                        variation_value_labels,
                        fixed_params_labels,
                        colors,
                        save_figure=False,
                        figure_name='frc_variations',
                        file_format='svg'):
    """
    Create a plot showing the variations in frequency response curves (FRCs)
    for different parameters.
    Inputs:
        data_list: List of parameter values for each variation (list of lists)
        freqs: List of frequency arrays for each variation (list of lists)
        amps: List of amplitude arrays for each variation (list of lists)
        variation_labels: Labels for the varying parameters (list of str)
        variation_value_labels: Labels for the values of the varying
            parameters (list of lists of str)
        fixed_params_labels: Labels for the fixed parameters (list of str)
        colors: Colors for each variation (list of str)
        save_figure: Flag to save the figure (bool, default=False)
        figure_name: Name for saving the figure (str, default='frc_variations')
        file_format: Format for saving the figure (str, default='svg')
    """
    plt.figure(figsize=(4, 3))
    groups = []
    for i, values in enumerate(data_list):
        group = [Line2D([], [], linestyle='none',
                        label=fixed_params_labels[i])]
        for j, var in enumerate(values):
            alpha = (j + 1) / len(values)
            plt.plot(
                freqs[i][j][0],
                amps[i][j][0],
                linestyle='-',
                color=colors[i],
                alpha=alpha
            )
            plt.plot(
                freqs[i][j][1],
                amps[i][j][1],
                linestyle='',
                marker='o',
                markerfacecolor='none',
                markeredgecolor=colors[i],
                markersize=3,
                alpha=alpha
            )
            group.append(Line2D([0], [0], color=colors[i], linestyle='-',
                                alpha=alpha,
                                label=variation_value_labels[i][j]))
        groups.append(group)

    plt.grid()
    plt.xlabel(r'Excitation Frequency $\Omega_{\mathrm{exc}}/' +
               r'\omega_{\mathrm{eig}}$')
    plt.ylabel(r'Response Amplitude $q_{\mathrm{exc}} / l_{\mathrm{beam}}$')
    plt.gca().set_box_aspect(.5)

    method_handles = [Line2D([0], [0], color='k', linestyle='-', label='AFT'),
                      Line2D([0], [0], color='k', linestyle='', marker='o',
                             markerfacecolor='none', markeredgecolor='k',
                             markersize=4, label='NN')]
    legend_method = plt.legend(handles=method_handles, loc='upper left')
    plt.gca().add_artist(legend_method)

    n = max(map(len, groups))
    empty = Line2D([], [], linestyle='none', label='')
    for group in groups:
        group += [empty] * (n - len(group))
    handles = [handle for group in groups for handle in group]
    plt.legend(
        handles=handles,
        ncol=len(groups),
        loc='upper right',
        columnspacing=1.5,
        handlelength=0.7,
        handletextpad=0.4,
        bbox_to_anchor=(1.4, -.4),
    )
    if save_figure:
        plt.savefig(f'figures/{figure_name}.{file_format}',
                    bbox_inches='tight')
