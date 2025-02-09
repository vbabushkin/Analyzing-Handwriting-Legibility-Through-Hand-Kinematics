import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utilities
import path_variables
from matplotlib.ticker import MaxNLocator

plt.style.use('default')
matplotlib.rcParams.update({'font.size': 17})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# Constants
custom_palette = sns.color_palette("Blues", 5)

def plot_distributions(expert):
    # load main data
    dfExpertLabels = pd.read_csv(path_variables.MAIN_DATA_FOLDER+"avg_legibility_score.csv")

    (X, Y, subjInfo) = utilities.load_multilabel_kinematics_data(path_variables.KINEMATICS_RAW_DATA, dfExpertLabels,
                                                                 True, 20)
    subjInfo = np.array(subjInfo)

    exp_subj_info = np.zeros((subjInfo.shape[0], 3))
    for k in range(1, 51):
        exp_subj_info[np.where(subjInfo[:, 0] == k)[0], 0:2] = subjInfo[np.where(subjInfo[:, 0] == k)[0], 0:2]
        exp_subj_info[np.where(subjInfo[:, 0] == k)[0], 2] = np.mean(
            subjInfo[np.where(subjInfo[:, 0] == k)[0], int(expert + 1)])

    Y = np.round(exp_subj_info[:, 2]).astype(int)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    labels_tr, counts_tr = np.unique(Y, return_counts=True)
    bar1 = ax1.bar(labels_tr, counts_tr, align='center')
    ax1.bar_label(bar1)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(labels_tr, labels_tr)
    ax1.set_xlabel("legibility score", fontsize=22)
    ax1.set_ylabel("number of samples", fontsize=22)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.margins(x=0.01)
    ax1.set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig(path_variables.FIGURES_PATH + "labels_distrib_expert_" + str(expert) + ".pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_distributions.py <expert>")
        sys.exit(1)

    expert_arg = int(sys.argv[1])
    plot_distributions(expert_arg)