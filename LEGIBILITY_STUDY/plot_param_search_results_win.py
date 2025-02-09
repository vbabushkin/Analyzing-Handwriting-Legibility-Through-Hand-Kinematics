import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import path_variables
import utilities

plt.style.use('default')
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# Constants
custom_palette = sns.color_palette("Blues", 5)
MIN_WIN = config.MIN_WIN
MAX_WIN = config.MAX_WIN
WIN_STEP = config.WIN_STEP
windows = np.arange(MIN_WIN, MAX_WIN, WIN_STEP)


def plot_param_search_results_win(expert):
    total_accuracy_array_per_expert = []
    for run in range(5):
        df = pd.read_csv(path_variables.RESULTS_PATH + "param_search_win_expert_" + str(expert) + "_run_" + str(
            run) + ".csv")
        tmp_acc_array_per_win_run = []
        for j in range(len(windows)):
            tmp_df = df.iloc[np.where(df["WINDOW"] == windows[j])[0], :]
            tmp_acc_array_per_win_run.append(tmp_df["ACC"].to_numpy())
        tmp_acc_array_per_win_run = np.squeeze(np.array(tmp_acc_array_per_win_run))
        total_accuracy_array_per_expert.append(np.mean(tmp_acc_array_per_win_run, axis=1))
    total_accuracy_array_per_expert = np.squeeze(np.array(total_accuracy_array_per_expert))
    total_accuracy_array = np.mean(total_accuracy_array_per_expert, axis=0)
    plt.figure(figsize=(9, 5))
    plt.plot(windows, total_accuracy_array, 'bo-')
    plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], [60, 65, 70, 75, 80, 85, 90])
    plt.title(f"Search for optimal window for expert {expert}")
    plt.xlabel("window size", fontsize=24)
    plt.ylabel("accuracy, %", fontsize=24)
    plt.xlim([0, 1800])
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_variables.FIGURES_PATH + f'param_search_win_5_folds_5_runs_expert_{expert}.pdf')
    plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_param_search_results_win.py <expert>")
        sys.exit(1)
    expert_arg = int(sys.argv[1])
    if utilities.all_results_win_search_exist(expert_arg):
        plot_param_search_results_win(expert_arg)
    else:
        print(
            f"csv files containing optimal window search results for a given expert {expert_arg} are missing in RESULTS folder. Please run param_search_win.py")
