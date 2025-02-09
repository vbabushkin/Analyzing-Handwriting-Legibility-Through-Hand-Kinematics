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
windows = [config.MIN_WIN,config.MED_WIN, (config.MAX_WIN // 64) * 64]

def plot_param_search_results_ovr(expert):
    total_accuracy_array_per_expert = [[], [], [], [], []]
    for run in range(5):
        df = pd.read_csv(path_variables.RESULTS_PATH + "param_search_ovr_expert_" + str(expert) + "_run_" + str(
            run ) + ".csv")
        tmp_acc_array_per_win_run = []
        for j in range(3):
            tmp_df = df.iloc[np.where(df["WINDOW"] == windows[j])[0], :]
            tmp_acc_array_per_win_run.append(tmp_df["ACC"].to_numpy())
        tmp_acc_array_per_win_run = np.squeeze(np.array(tmp_acc_array_per_win_run))
        total_accuracy_array_per_expert[run].append(tmp_acc_array_per_win_run)
    total_accuracy_array_per_expert = np.squeeze(np.array(total_accuracy_array_per_expert))

    # average by overlaps
    total_accuracy_array = np.mean(total_accuracy_array_per_expert, axis=0)
    x = np.arange(0, 100, 10)
    accuracy_per_win = [[], [], []]
    for i in range(3):
        a = total_accuracy_array[i, :]
        for ovr in np.arange(0, 50, 5):
            accuracy_per_win[i].append(np.mean(a[ovr:ovr + 5]))
    accuracy_per_win = np.array(accuracy_per_win)
    y1 = accuracy_per_win[0, :]
    y2 = accuracy_per_win[1, :]
    y3 = accuracy_per_win[2, :]
    plt.plot(x, y1, 'r.-', label='64')
    plt.plot(x, y2, 'g^-', label='896')
    plt.plot(x, y3, 'bs-', label='1728')
    plt.xticks(x, x)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [40, 50, 60, 70, 80, 90])
    plt.xlabel("overlap, %", fontsize=24)
    plt.ylabel("accuracy, %", fontsize=24)
    plt.legend(title="window", fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_variables.FIGURES_PATH + 'expert_' + str(expert) + '_overlap_5_folds_5_runs.pdf')
    plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_param_search_results_ovr.py <expert>")
        sys.exit(1)

    expert_arg = int(sys.argv[1])
    if utilities.all_results_overlap_search_exist(expert_arg):
        plot_param_search_results_ovr(expert_arg)
    else:
        print(
            f"csv files containing optimal overlap search results for a given expert {expert_arg} are missing in RESULTS folder. Please run param_search_ovr.py")
